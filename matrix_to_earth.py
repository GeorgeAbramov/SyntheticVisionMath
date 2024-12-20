import numpy as np

def get_rotation_matrix(roll: float, pitch: float, heading: float) -> np.ndarray:
    """
    Матрица перехода из связанной в земную систему координат
    
    Args:
        roll (float): угол крена в радианах
        pitch (float): угол тангажа в радианах
        heading (float): угол курса в радианах
        
    Returns:
        np.ndarray: матрица поворота 3x3
    """
    # Матрица поворота вокруг OX (крен)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Матрица поворота вокруг OY (тангаж)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Матрица поворота вокруг OZ (курс)
    R_z = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    
    # Полная матрица поворота (порядок: крен -> тангаж -> курс)
    R = R_z @ R_y @ R_x
    return R