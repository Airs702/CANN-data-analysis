import numpy as np
import math
import random
from scipy.special import i0
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# === 常量定义===
NB_ROI= 16
NB_BUMP_MAX = 4
SIGMA_DIFF = 0.5
AMPLI_MIN = 2.0
KAPPA_MEAN = 2.5
SIG2 = 1.0
SIGCOUP = 2 * math.pi / NB_ROI
SIGCOUP2 = SIGCOUP ** 2
PENBUMP = 4.0
JC = 1.800
BETA = 5.0
NB_MOVES = 20000


# === 快速选择算法===
def quick_select(arr, k):
    """快速选择算法实现"""
    arr = arr.copy()
    left, right = 0, len(arr) - 1
    while left <= right:
        pivot_idx = random.randint(left, right)
        pivot_val = arr[pivot_idx]
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

        store_idx = left
        for i in range(left, right):
            if arr[i] < pivot_val:
                arr[i], arr[store_idx] = arr[store_idx], arr[i]
                store_idx += 1

        arr[store_idx], arr[right] = arr[right], arr[store_idx]

        if store_idx == k:
            return arr[store_idx]
        elif store_idx < k:
            left = store_idx + 1
        else:
            right = store_idx - 1
    return arr[k]


def median_via_quickselect(arr):
    """使用快速选择计算中位数"""
    n = len(arr)
    if n % 2 == 1:
        return quick_select(arr, n // 2)
    else:
        left = quick_select(arr, n // 2 - 1)
        right = quick_select(arr, n // 2)
        return 0.5 * (left + right)


# === 数据结构===
class SiteBump:
    def __init__(self):
        self.nbump = 0
        self.pos = []
        self.ampli = []
        self.kappa = []
        self.logl = 0.0

    def clone(self):
        """创建对象的深拷贝"""
        new = SiteBump()
        new.nbump = self.nbump
        new.pos = self.pos.copy()
        new.ampli = self.ampli.copy()
        new.kappa = self.kappa.copy()
        new.logl = self.logl
        return new


# === 核心函数（匹配C版本）===
def vonmises(x, mu, kappa):
    """冯·米塞斯分布"""
    return math.exp(kappa * math.cos(x - mu)) / (2 * math.pi * i0(kappa))


def interf_logl(b1, b2):
    """时间点间耦合似然"""
    # 使用固定大小的检查数组（匹配C版本）
    used1 = [0] * NB_BUMP_MAX
    used2 = [0] * NB_BUMP_MAX
    logli = 0.0

    # 计算最大链接数（匹配C版本）
    max_links = min(b1.nbump, b2.nbump, NB_BUMP_MAX)

    for _ in range(max_links):
        min_diff = 2 * math.pi
        found = False
        i_best = j_best = -1

        # 遍历b1的峰
        for i in range(b1.nbump):
            if used1[i]:
                continue
            # 遍历b2的峰
            for j in range(b2.nbump):
                if used2[j]:
                    continue
                # 计算圆周距离
                dist = abs(b1.pos[i] - b2.pos[j])
                diff = min(dist, 2 * math.pi - dist)

                if diff < min_diff:
                    min_diff = diff
                    i_best, j_best = i, j
                    found = True

        if found:
            used1[i_best] = 1
            used2[j_best] = 1
            logli += math.exp(-0.5 * min_diff ** 2 / SIGCOUP2)

    return BETA * JC * logli


def site_logl(intens, bump):
    """单个时间点似然"""
    logl = -bump.nbump * PENBUMP
    for i in range(NB_ROI):
        x = i * 2 * math.pi / NB_ROI
        val = 0.0
        # 计算每个峰在当前ROI的贡献
        for j in range(bump.nbump):
            val += bump.ampli[j] * vonmises(x, bump.pos[j], bump.kappa[j])

        diff = intens[i] - val
        logl -= 0.5 * diff ** 2 / SIG2

    return BETA * logl


def diffuse(bump):
    """扩散峰位置"""
    if bump.nbump == 0:
        return True
    i = random.randrange(bump.nbump)
    new_pos = bump.pos[i] + random.gauss(0, SIGMA_DIFF)
    new_pos %= 2 * math.pi
    if new_pos < 0:
        new_pos += 2 * math.pi
    bump.pos[i] = new_pos
    return False


def change_ampli(bump):
    """改变峰振幅"""
    if bump.nbump == 0:
        return True
    i = random.randrange(bump.nbump)
    delta = random.gauss(0, 1.0)
    if bump.ampli[i] + delta <= 0:
        return True
    bump.ampli[i] += delta
    return False


def change_width(bump):
    """改变峰宽度"""
    if bump.nbump == 0:
        return True
    i = random.randrange(bump.nbump)
    new_kappa = bump.kappa[i] + random.gauss(0, 0.5)
    if new_kappa < 2.0 or new_kappa > 6.0:
        return True
    bump.kappa[i] = new_kappa
    return False


def create_bump(bump):
    """创建新峰"""
    if bump.nbump >= NB_BUMP_MAX:
        return True
    bump.pos.append(random.uniform(0, 2 * math.pi))
    bump.ampli.append(AMPLI_MIN)
    bump.kappa.append(KAPPA_MEAN)
    bump.nbump += 1
    return False


def del_bump(bump):
    """删除峰"""
    if bump.nbump == 0:
        return True
    i = random.randrange(bump.nbump)
    # 用最后一个元素覆盖
    if i < bump.nbump - 1:
        bump.pos[i] = bump.pos[-1]
        bump.ampli[i] = bump.ampli[-1]
        bump.kappa[i] = bump.kappa[-1]
    # 移除最后一个元素
    bump.pos.pop()
    bump.ampli.pop()
    bump.kappa.pop()
    bump.nbump -= 1
    return False


# === MCMC主函数（性能优化）===
def mcmc(trial):
    """全局MCMC优化（修复性能问题）"""
    ntime = trial['nbt']
    # 初始化所有时间点的峰状态
    bumps = [SiteBump() for _ in range(ntime)]
    interfe = [0.0] * (ntime - 1)

    # 初始似然计算
    total_logl = 0.0
    for i in range(ntime):
        data_seg = trial['data'][i * NB_ROI:(i + 1) * NB_ROI]
        bumps[i].logl = site_logl(data_seg, bumps[i])
        total_logl += bumps[i].logl

    for i in range(ntime - 1):
        interfe[i] = interf_logl(bumps[i], bumps[i + 1])
        total_logl += interfe[i]

    print(f"初始似然: {total_logl:.2f}")

    # MCMC迭代
    start_time = time.time()
    for move in range(NB_MOVES):
        for j in range(ntime):
            current = bumps[j]
            proposal = current.clone()

            # 选择操作类型
            rand_val = random.random()
            operation_failed = True

            if rand_val < 0.01:
                operation_failed = create_bump(proposal)
            elif rand_val < 0.01 * (1 + proposal.nbump):
                operation_failed = del_bump(proposal)
            elif rand_val < 0.3:
                operation_failed = diffuse(proposal)
            elif rand_val < 0.4:
                operation_failed = change_width(proposal)
            else:
                operation_failed = change_ampli(proposal)

            # 如果操作成功（没有失败）
            if not operation_failed:
                # 计算新状态的局部似然
                data_seg = trial['data'][j * NB_ROI:(j + 1) * NB_ROI]
                loglt = site_logl(data_seg, proposal)

                # 计算耦合变化
                delta_logl = loglt - current.logl

                # 处理边界情况
                if j == 0:  # 第一个时间点
                    loglit1 = interf_logl(proposal, bumps[1])
                    delta_logl += loglit1 - interfe[0]
                elif j == ntime - 1:  # 最后一个时间点
                    loglit1 = interf_logl(bumps[j - 1], proposal)
                    delta_logl += loglit1 - interfe[j - 1]
                else:  # 中间时间点
                    loglit1 = interf_logl(bumps[j - 1], proposal)
                    loglit2 = interf_logl(proposal, bumps[j + 1])
                    delta_logl += (loglit1 - interfe[j - 1]) + (loglit2 - interfe[j])

                # Metropolis-Hastings接受准则
                if delta_logl > 0 or random.random() < math.exp(delta_logl):
                    proposal.logl = loglt
                    bumps[j] = proposal

                    # 更新耦合项
                    if j == 0:
                        interfe[0] = loglit1
                    elif j == ntime - 1:
                        interfe[j - 1] = loglit1
                    else:
                        interfe[j - 1] = loglit1
                        interfe[j] = loglit2

                    total_logl += delta_logl

        # 显示进度
        if move % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"迭代 {move}/{NB_MOVES}, 当前似然: {total_logl:.2f}, 耗时: {elapsed:.1f}秒")

    print(f"MCMC完成, 总耗时: {time.time() - start_time:.2f}秒")
    return bumps


# === 输出函数===
def save_outputs(bumps, trial, output_prefix="bump"):
    """输出三种结果文件"""
    # 1. 输出fits.dat
    with open(f"{output_prefix}-fits.dat", "w") as f_fits:
        for t, bump in enumerate(bumps):
            for i in range(bump.nbump):
                f_fits.write(f"{t} {bump.pos[i]:.6f} {bump.ampli[i]:.6f} {bump.kappa[i]:.6f}\n")

    # 2. 输出nbump.dat
    with open(f"{output_prefix}-nbump.dat", "w") as f_nbump:
        for t, bump in enumerate(bumps):
            f_nbump.write(f"{t} {bump.nbump}")
            for i in range(NB_ROI):
                x = i * 2 * math.pi / NB_ROI
                val = 0.0
                for j in range(bump.nbump):
                    val += bump.ampli[j] * vonmises(x, bump.pos[j], bump.kappa[j])
                f_nbump.write(f" {val:.6f}")
            f_nbump.write("\n")

    # 3. 输出centrbump.dat（只输出2列）
    with open(f"{output_prefix}-centrbump.dat", "w") as f_centr:
        for t, bump in enumerate(bumps):
            # 获取当前时间点的原始数据
            data_segment = trial['data'][t * NB_ROI:(t + 1) * NB_ROI]

            for i in range(bump.nbump):
                for roi in range(NB_ROI):
                    roi_pos = roi * 2 * math.pi / NB_ROI
                    # 计算距离并调整到[-π, π]
                    dist = bump.pos[i] - roi_pos
                    if dist > math.pi:
                        dist -= 2 * math.pi
                    elif dist < -math.pi:
                        dist += 2 * math.pi

                    # 计算归一化振幅（原始数据点/峰振幅）
                    norm_amp = data_segment[roi] / bump.ampli[i]

                    f_centr.write(f"{dist:.6f} {norm_amp:.6f}\n")



def generate_1DCANN_bump(
    data_path,
    output_path="bump_cann_custom.gif",
    max_height_value=0.5,
    max_width_range=40,
    npoints=300,
    nframes=None,
    fps=5
):
    # ==== 平滑函数 ====
    def smooth(x, window=3):
        return np.convolve(x, np.ones(window)/window, mode='same')

    def smooth_circle(values, window=5):
        pad = window // 2
        values_ext = np.concatenate([values[-pad:], values, values[:pad]])
        kernel = np.ones(window) / window
        smoothed = np.convolve(values_ext, kernel, mode='valid')
        return smoothed

    # ==== 读取数据 ====
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.loadtxt(data_path)

    if nframes is None or nframes > len(data):
        nframes = len(data)

    positions_raw = data[:nframes, 1]
    heights_raw = data[:nframes, 2]
    widths_raw = data[:nframes, 3]

    # ==== 平滑处理 ====
    positions = smooth(positions_raw, window=3)
    heights_raw_smooth = smooth(heights_raw, window=3)
    widths_raw_smooth = smooth(widths_raw, window=3)

    # ==== 参数设定 ====
    theta = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
    base_radius = 1.0

    # ==== 归一化处理 ====
    min_height, max_height = np.min(heights_raw_smooth), np.max(heights_raw_smooth)
    heights = np.interp(heights_raw_smooth, (min_height, max_height), (0.1, max_height_value))

    min_width, max_width = np.min(widths_raw_smooth), np.max(widths_raw_smooth)
    width_ranges = np.interp(widths_raw_smooth, (min_width, max_width), (2, max_width_range)).astype(int)

    # ==== 初始化图形 ====
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    line, = ax.plot([], [], color='red', linewidth=2)

    def init():
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.axis('off')
        return line,

    def update(frame):
        pos_angle = positions[frame]
        height = heights[frame]
        width_range = width_ranges[frame]

        center_idx = np.argmin(np.abs(theta - pos_angle))
        r = np.ones(npoints) * base_radius

        max_kernel_size = 60
        sigma = width_range / 2

        for offset in range(-max_kernel_size, max_kernel_size + 1):
            dist = abs(offset)
            gauss_weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
            if gauss_weight < 0.01:
                continue
            idx = (center_idx + offset) % npoints
            r[idx] += height * gauss_weight

        r = smooth_circle(r, window=5)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        ax.clear()
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.axis('off')

        # 单位圆基准线
        inner_x = base_radius * np.cos(theta)
        inner_y = base_radius * np.sin(theta)
        ax.plot(inner_x, inner_y, color='gray', linestyle='--', linewidth=1)

        # bump 曲线
        ax.plot(x, y, color='red', linewidth=2)

        # bump 中心点位置
        dot_radius = base_radius * 0.96
        center_x = dot_radius * np.cos(pos_angle)
        center_y = dot_radius * np.sin(pos_angle)
        ax.plot(center_x, center_y, 'o', color='black', markersize=6)

        return line,

    ani = FuncAnimation(fig, update, frames=nframes, init_func=init, blit=False)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    print(f"✅ Animation saved to: {output_path}")
