
import numpy as np
import os
import re
import datetime

# 追加记录功能的辅助常量与函数
BASE_PATH = "/home/stuwork/MRPC-2025-homework"
SO3_CTRL_SRC = f"{BASE_PATH}/code/src/quadrotor_simulator/so3_control/src/so3_control_nodelet.cpp"
RUN_HISTORY_PATH = f"{BASE_PATH}/solutions/run_history.txt"

def generate_run_id():
    """生成唯一运行 ID（时间戳）。"""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def read_kx_kv_from_source():
    """
    从 so3_control_nodelet.cpp 第 103、104 行读取 kx/kv。
    约定这两行始终是：
      kx_ = Eigen::Vector3d(...);
      kv_ = Eigen::Vector3d(...);
    """
    try:
        with open(SO3_CTRL_SRC, "r") as f:
            lines = f.readlines()
        # 直接按固定行号提取
        kx_line = lines[102].strip() if len(lines) > 102 else ""
        kv_line = lines[103].strip() if len(lines) > 103 else ""

        def parse_vec(line):
            m = re.search(r"Eigen::Vector3d\\(([^)]*)\\)", line)
            return [float(x.strip()) for x in m.group(1).split(",")] if m else None

        kx = parse_vec(kx_line)
        kv = parse_vec(kv_line)
        return kx, kv
    except Exception:
        return None, None

def append_run_history(run_id, kx, kv, rmse, total_time, total_length, additional_score, overall_score):
    """将本次运行结果追加到历史记录文件。"""
    kx_str = ", ".join(f"{v:.4f}" for v in kx) if kx else "N/A"
    kv_str = ", ".join(f"{v:.4f}" for v in kv) if kv else "N/A"
    line = (
        f"ID={run_id} "
        f"kx=[{kx_str}] kv=[{kv_str}] "
        f"{rmse} {total_time} {total_length} {additional_score} {overall_score}\n"
    )
    with open(RUN_HISTORY_PATH, "a") as f:
        f.write(line)
def calculate_rmse_and_more():
    des_pos_data = []
    pos_data = []
    time_stamps = []
    time_stamps2 = []
    length_increments = []

    with open("/home/stuwork/MRPC-2025-homework/code/src/quadrotor_simulator/so3_control/src/control_data.txt", "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            data = line.strip().split()
            des_pos = np.array([float(data[1]), float(data[2]), float(data[3])])
            pos = np.array([float(data[4]), float(data[5]), float(data[6])])
            des_pos_data.append(des_pos)
            pos_data.append(pos)
            time_stamps.append(float(data[0]))

            if i > 0:
                # 计算当前时刻与上一时刻实际运行位置的差值作为轨迹长度增量
                prev_pos = np.array([float(lines[i - 1].strip().split()[4]),
                                    float(lines[i - 1].strip().split()[5]),
                                    float(lines[i - 1].strip().split()[6])])
                length_increment = np.linalg.norm(pos - prev_pos)
                length_increments.append(length_increment)

    des_pos_array = np.array(des_pos_data)
    pos_array = np.array(pos_data)
    time_stamps_array = np.array(time_stamps)
    length_increments_array = np.array(length_increments)

    if des_pos_array.shape!= pos_array.shape:
        raise ValueError("期望位置数据和实际位置数据的形状不一致。")
    
    with open("/home/stuwork/MRPC-2025-homework/code/src/quadrotor_simulator/so3_control/src/control_timedata.txt", "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            datatime = line.strip().split()
            time_stamps2.append(float(datatime[0]))
    
    time_stamps2_array = np.array(time_stamps2)

    # 计算均方根误差（RMSE）
    diff_array = des_pos_array - pos_array
    squared_diff_array = diff_array ** 2
    mean_squared_error = np.mean(squared_diff_array)
    rmse = np.sqrt(mean_squared_error)

    # 计算轨迹运行总时间
    total_time = time_stamps2_array[-1] - time_stamps_array[0]

    # 计算总轨迹长度
    total_length = np.sum(length_increments_array)

    # 检查是否发生了碰撞
    additional_score = check_additional_file()

    overall_score = 200. * rmse + 1./5. * total_time + 1./5. * total_length + 40. * additional_score
    
    return rmse, total_time, total_length, additional_score, overall_score
def check_additional_file():
    file_path = "/home/stuwork/MRPC-2025-homework/code/src/quadrotor_simulator/so3_control/src/issafe.txt"  # 替换为实际的文件路径
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()
            if content:
                return 1
            return 0
    except FileNotFoundError:
        return 0

if __name__ == "__main__":
    try:
        rmse, total_time, total_length, additional_score, overall_score = calculate_rmse_and_more()
        print(f"计算得到的均方根误差（RMSE）值为: {rmse}")
        print(f"轨迹运行总时间为: {total_time}")
        print(f"总轨迹长度为: {total_length}")
        print(f"是否发生了碰撞: {additional_score}")
        print(f"综合评价得分为(综合分数越低越好): {overall_score}")
        
        result_file_path = "/home/stuwork/MRPC-2025-homework/solutions/result.txt"
        with open(result_file_path, "w") as f:
            f.write(f"{rmse} {total_time} {total_length} {additional_score} {overall_score}")

        # 新增：记录运行历史
        run_id = generate_run_id()
        kx, kv = read_kx_kv_from_source()
        append_run_history(run_id, kx, kv, rmse, total_time, total_length, additional_score, overall_score)
    except ValueError as e:
        print(f"发生错误: {e}")
        
