Вариант с денойзингом.

1) Для обучения и сохранения модели выполнить команду `python train.py`.
2) Для конвертации модели в удобное представление выполнить команду `python convert.py -fuse_relu <0 или 1>`.
3) `mkdir build && cd build`
   `cmake .. -DBLOCK_SIZE=<X>`
   `make all`
   `./cuda_denoise <путь/к/изображению> -benchmark <кол-во запусков>.`

Система: 
    Процессор: Intel(R) Core(TM) i7-12700K
    GPU: Nvidia GeForce RTX 3060

Замеры:
    500 итераций на изображении 28х28 со слиянием ReLU: 3.07мс в среднем.
    500 итераций на изображении 28х28 без слияния ReLU: 3.43мс в среднем.