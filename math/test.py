import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from trajectory_prediction import TrajectoryPredictor
from terrain_checker import TerrainChecker

def load_initial_state(file_path: Path) -> np.ndarray:
    """Загрузка начального состояния из файла"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Пропускаем заголовок и берем первую строку данных
        data = lines[1].strip().split()  # берем вторую строку и разбиваем по пробелам
        
        try:
            # Преобразуем строки в числа
            roll = float(data[0])
            pitch = float(data[1])
            heading = float(data[2])
            vx = float(data[3])
            vy = float(data[4])
            vz = float(data[5])
            lat = float(data[6])
            lon = float(data[7])
            alt = float(data[8])
            
            print(f"\nЗагруженные данные:")
            print(f"Углы: roll={roll}°, pitch={pitch}°, heading={heading}°")
            print(f"Скорости: Vx={vx}, Vy={vy}, Vz={vz}")
            print(f"Координаты: lat={lat}°, lon={lon}°, alt={alt}м")
            
            # Формируем вектор состояния
            return np.array([
                np.radians(lat),      # широта в радианах
                np.radians(lon),      # долгота в радианах
                alt,                  # высота в метрах
                vx,                   # скорость по x
                vy,                   # скорость по y
                vz,                   # скорость по z
                np.radians(roll),     # крен в радианах
                np.radians(pitch),    # тангаж в радианах
                np.radians(heading)   # курс в радианах
            ])
            
        except (IndexError, ValueError) as e:
            print(f"\nОшибка при чтении данных:")
            print(f"Строка данных: {data}")
            print(f"Ошибка: {str(e)}")
            raise

def print_trajectory_point(point: np.ndarray, t: float, is_initial: bool = False):
    """Вывод информации о точке траектории"""
    point_type = "Начальная" if is_initial else f"t={t}с"
    print(f"\n{point_type} точка:")
    print(f"Координаты:")
    print(f"  lat={np.degrees(point[0])}°")
    print(f"  lon={np.degrees(point[1])}°")
    print(f"  alt={point[2]}м")
    print(f"Скорости:")
    print(f"  Vx={point[3]} м/с")
    print(f"  Vy={point[4]} м/с")
    print(f"  Vz={point[5]} м/с")
    print(f"Углы:")
    print(f"  roll={np.degrees(point[6])}°")
    print(f"  pitch={np.degrees(point[7])}°")
    print(f"  heading={np.degrees(point[8])}°")

def predict_from_point(predictor: TrajectoryPredictor, nav_data: np.ndarray, point_index: int, terrain_checker: TerrainChecker):
    """
    Прогноз траектории от конкретной точки навигационных данных
    
    Args:
        predictor: предиктор траектории
        nav_data: массив навигационных данных
        point_index: индекс текущей точки
    """
    # Получаем текущее состояние из навигационных данных
    current_data = nav_data[point_index]
    
    # Формируем начальное состояние для прогноза
    initial_state = np.array([
        np.radians(current_data[6]),  # lat
        np.radians(current_data[7]),  # lon
        current_data[8],              # alt
        current_data[3],              # Vx
        current_data[4],              # Vy
        current_data[5],              # Vz
        np.radians(current_data[0]),  # roll
        np.radians(current_data[1]),  # pitch
        np.radians(current_data[2])   # heading
    ])
    
    # Прогноз на 60 секунд вперед с шагом 0.1 секунды
    trajectory = predictor.predict_trajectory(
        initial_state=initial_state,
        dt=0.01,  # уменьшаем шаг до 0.1 секунды для более точного прогноза
        prediction_time=60.0
    )
    
    # Проверяем каждую точку траектории на столкновение с рельефом
    for i, point in enumerate(trajectory):
        has_collision, distance = terrain_checker.check_collision(point)
        warning_level = terrain_checker.get_collision_warning_level(distance)
        
        if warning_level > 0:
            time = i * 0.1  # время в секундах
            print(f"\nПРЕДУПРЕЖДЕНИЕ! Уровень опасности: {warning_level}")
            print(f"Время до потенциального столкновения: {time:.1f}с")
            print(f"Расстояние до рельефа: {distance:.2f}м")
            
        if has_collision:
            print(f"\nВНИМАНИЕ! Обнаружено пересечение с рельефом!")
            break
    
    return trajectory

def analyze_flight_data():
    """Анализ полетных данных с прогнозом для каждой точки"""
    data_path = Path(__file__).parent / 'navData.txt'
    
    # Инициализация проверки рельефа с локальным файлом .mbtiles
    terrain_checker = TerrainChecker()  # Теперь без API ключа
    
    # Загружаем все навигационные данные
    nav_data = np.loadtxt(data_path, skiprows=1)
    predictor = TrajectoryPredictor()
    
    # Для каждой точки делаем прогноз
    for i in range(len(nav_data)):
        print(f"\n=== Точка {i} ===")
        print("Текущие параметры:")
        print(f"Координаты: lat={nav_data[i,6]}°, lon={nav_data[i,7]}°, alt={nav_data[i,8]}м")
        print(f"Скорости: Vx={nav_data[i,3]}, Vy={nav_data[i,4]}, Vz={nav_data[i,5]}")
        print(f"Углы: roll={nav_data[i,0]}°, pitch={nav_data[i,1]}°, heading={nav_data[i,2]}°")
        
        # Прогноз от текущей точки
        predicted_trajectory = predict_from_point(predictor, nav_data, i, terrain_checker)
        
        # Выводим прогноз через 60 секунд
        final_point = predicted_trajectory[-1]
        print("\nПрогноз через 60 секунд:")
        print(f"Координаты: lat={np.degrees(final_point[0])}°, lon={np.degrees(final_point[1])}°, alt={final_point[2]}м")
        print(f"Скорости: Vx={final_point[3]}, Vy={final_point[4]}, Vz={final_point[5]}")
        print(f"Углы: roll={np.degrees(final_point[6])}°, pitch={np.degrees(final_point[7])}°, heading={np.degrees(final_point[8])}°")
        
        # Вычисляем изменения
        d_lat = np.degrees(final_point[0]) - nav_data[i,6]
        d_lon = np.degrees(final_point[1]) - nav_data[i,7]
        d_alt = final_point[2] - nav_data[i,8]
        
        print("\nИзменения через 60 секунд:")
        print(f"Δlat: {d_lat}°")
        print(f"Δlon: {d_lon}°")
        print(f"Δalt: {d_alt}м")

if __name__ == "__main__":
    analyze_flight_data() 