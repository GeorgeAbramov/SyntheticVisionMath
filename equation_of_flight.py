import numpy as np
from matrix_to_earth import get_rotation_matrix

class FlightEquations:
    """
    Уравнения движения летательного аппарата
    """
    def __init__(self):
        self.g = 9.81  # ускорение свободного падения
        self.R_earth = 6371000.0  # радиус Земли в метрах
        
    def calculate_derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Расчет производных состояния ЛА
        
        Args:
            state: вектор состояния [φ, λ, h, Vx, Vy, Vz, γ, θ, ψ]
                φ - широта
                λ - долгота
                h - высота
                Vx, Vy, Vz - скорости в связанной СК
                γ - крен
                θ - тангаж
                ψ - курс
            t: текущее время
            
        Returns:
            np.ndarray: вектор производных состояния
        """
        # Распаковка состояния
        lat, lon, alt = state[0:3]
        v_body = state[3:6]
        roll, pitch, heading = state[6:9]
        
        # Пересчет скоростей из связанной в земную СК
        R = get_rotation_matrix(roll, pitch, heading)
        v_earth = R @ v_body
        
        # Масштабные коэффициенты для географических координат
        lat_scale = 1.0 / (self.R_earth + alt)
        lon_scale = 1.0 / ((self.R_earth + alt) * np.cos(lat))
        
        # Производные координат
        d_lat = v_earth[0] * lat_scale
        d_lon = v_earth[1] * lon_scale
        d_alt = v_earth[2]
        
        # Ускорения в земной СК
        g_earth = np.array([0, 0, -self.g])
        
        # Пересчет ускорений в связанную СК
        d_v_body = R.T @ g_earth
        
        # Производные углов (упрощенная модель демпфирования)
        d_roll = -0.5 * roll  # демпфирование крена
        d_pitch = -0.3 * pitch  # стремление к горизонту
        
        # Курс по направлению движения
        v_horiz = np.sqrt(v_body[0]**2 + v_body[1]**2)
        if v_horiz > 1e-6:
            target_heading = np.arctan2(v_body[1], v_body[0])
            heading_diff = target_heading - heading
            heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
            d_heading = 0.2 * heading_diff
        else:
            d_heading = 0
        
        return np.array([
            d_lat, d_lon, d_alt,           # изменение координат
            d_v_body[0], d_v_body[1], d_v_body[2],  # изменение скоростей
            d_roll, d_pitch, d_heading     # изменение углов
        ])
    

         
        
