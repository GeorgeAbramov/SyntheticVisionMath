import sqlite3
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from io import BytesIO
import mercantile
from pathlib import Path

class TerrainChecker:
    def __init__(self, mbtiles_path: str = None):
        """
        Инициализация проверки рельефа
        
        Args:
            mbtiles_path: путь к файлу .mbtiles
        """
        if mbtiles_path is None:
            mbtiles_path = str(Path(__file__).parent / 'maptiler-osm-2020-02-10-v3.11-europe_russia-european-part.mbtiles')
        
        self.conn = sqlite3.connect(mbtiles_path)
        self.cursor = self.conn.cursor()
        
        # Проверяем структуру базы данных
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        print("Доступные таблицы:", [table[0] for table in tables])
        
        # Для первой таблицы выводим структуру
        if tables:
            self.cursor.execute(f"PRAGMA table_info({tables[0][0]})")
            columns = self.cursor.fetchall()
            print(f"\nСтруктура таблицы {tables[0][0]}:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Выводим первую строку данных
            self.cursor.execute(f"SELECT * FROM {tables[0][0]} LIMIT 1")
            first_row = self.cursor.fetchone()
            print("\nПример данных:", first_row)
    
    def __del__(self):
        """Закрытие соединения с базой данных"""
        if hasattr(self, 'conn'):
            self.conn.close()
            
    def _get_tile_data(self, x: int, y: int, z: int) -> Optional[bytes]:
        """
        Получение данных тайла из базы
        
        Args:
            x, y: координаты тайла
            z: уровень зума
            
        Returns:
            bytes: данные PNG тайла или None
        """
        try:
            # В .mbtiles координата y инвертирована
            y = (1 << z) - 1 - y
            
            # Сначала проверим правильное имя таблицы и столбцов
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in self.cursor.fetchall()]
            
            # Попробуем найти таблицу с тайлами
            tile_table = None
            for table in tables:
                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in self.cursor.fetchall()]
                if any('tile' in col.lower() for col in columns):
                    tile_table = table
                    break
            
            if tile_table:
                # Выводим запрос, который собираемся выполнить
                query = f"SELECT * FROM {tile_table} WHERE zoom_level=? AND tile_column=? AND tile_row=?"
                print(f"Выполняем запрос: {query} с параметрами ({z}, {x}, {y})")
                
                self.cursor.execute(query, (z, x, y))
                row = self.cursor.fetchone()
                if row:
                    # Определяем индекс столбца с данными тайла
                    self.cursor.execute(f"PRAGMA table_info({tile_table})")
                    columns = [col[1] for col in self.cursor.fetchall()]
                    tile_data_index = next(i for i, col in enumerate(columns) if 'tile' in col.lower())
                    return row[tile_data_index]
            return None
        except Exception as e:
            print(f"Ошибка при получении тайла: {e}")
            return None
            
    def _get_tile_coords(self, lat: float, lon: float, zoom: int = 14) -> Tuple[int, int, int]:
        """Получение координат тайла для заданной точки"""
        tile = mercantile.tile(lon, lat, zoom)
        return tile.x, tile.y, zoom
        
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        Получение высоты рельефа для заданных координат
        
        Args:
            lat: широта в градусах
            lon: долгота в градусах
            
        Returns:
            Optional[float]: высота рельефа в метрах или None
        """
        try:
            # Получаем координаты тайла
            x, y, z = self._get_tile_coords(lat, lon)
            
            # Получаем данные тайла
            tile_data = self._get_tile_data(x, y, z)
            if tile_data is None:
                return None
                
            # Открываем изображение тайла
            img = Image.open(BytesIO(tile_data))
            
            # Получаем координаты пикселя внутри тайла
            bbox = mercantile.bounds(x, y, z)
            px = int((lon - bbox.west) / (bbox.east - bbox.west) * img.width)
            py = int((lat - bbox.north) / (bbox.south - bbox.north) * img.height)
            
            # Получаем значения RGB
            r, g, b = img.getpixel((px, py))
            
            # Декодируем высоту
            elevation = -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1)
            return float(elevation)
            
        except Exception as e:
            print(f"Ошибка при получении высоты рельефа: {e}")
            return None
            
    def check_collision(self, trajectory_point: np.ndarray) -> Tuple[bool, float]:
        """
        Проверка на столкновение с рельефом
        
        Returns:
            Tuple[bool, float]: (есть_столкновение, расстояние_до_рельефа)
        """
        lat = np.degrees(trajectory_point[0])
        lon = np.degrees(trajectory_point[1])
        alt = trajectory_point[2]
        
        # Проверяем, находимся ли мы в пределах европейской части России
        if not (41.81104 <= lat <= 81.47299 and 19.3951 <= lon <= 46.43616):
            print(f"Предупреждение: координаты ({lat}, {lon}) вне зоны покрытия карты")
            return False, float('inf')
        
        terrain_height = self.get_elevation(lat, lon)
        if terrain_height is None:
            return False, float('inf')
            
        distance_to_terrain = alt - terrain_height
        return distance_to_terrain < 0, distance_to_terrain
    
    def get_collision_warning_level(self, distance: float) -> int:
        """
        Определение уровня предупреждения о столкновении
        
        Returns:
            int: 0 - нет угрозы, 1 - низкий, 2 - средний, 3 - высокий
        """
        if distance > 1000:  # более 1000м до рельефа
            return 0
        elif distance > 500:  # от 500м до 1000м
            return 1
        elif distance > 200:  # от 200м до 500м
            return 2
        else:  # менее 200м
            return 3