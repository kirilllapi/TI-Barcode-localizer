# Порядок настройки и запуска.


https://github.com/user-attachments/assets/1a03d481-c32a-465c-a6e3-8551618a4718


## Получение кадров с Luck-fox Pico Ultra W:
  1) Прошить Luck-fox Pico Ultra W под BuildRoot (Документация: https://wiki.luckfox.com/Luckfox-Pico/Luckfox-Pico-RV1106/Luckfox-Pico-Ultra-W/Luckfox-Pico-emmc-burn-image/);
  2) Включить драйвера камеры (CSI);
  3) Подключить плату по USB к компьютеру;
  4) Запустить ```main.py```.

### Рекомендуется использовать CUDA (некоторые операции всё равно выполняются на CPU)
*Cuda работает через раз.
#### Версия Python: ```3.12```

Гайд по установке CUDA: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls
#### Cuda:
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network

#### cuDNN:
https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

#### Системные переменные:

![image](https://github.com/user-attachments/assets/d21de4dc-c5e7-427f-bc66-7f11951fdbb1)
`

#### Если использовать CUDA:
```
# Загрузка модели с использованием CUDA
cuda_path = os.path.join(os.environ["CUDA_PATH"], "bin")
onr.preload_dlls(cuda=True, cudnn=False, directory=cuda_path)

cudnn_path = os.path.join(os.environ["CUDNN_PATH"], "bin\\12.9")
onr.preload_dlls(cuda=False, cudnn=True, directory=cudnn_path)

onr.print_debug_info()

session  = onr.InferenceSession("source/model/model.onnx", providers=['CUDAExecutionProvider'])
```

#### Если использовать только CPU:
```
# Загрузка модели с использованием CPU
session  = onr.InferenceSession("source/model/model.onnx", providers=['CPUExecutionProvider'])
```

# При запуске использовались

CPU:

	13th Gen Intel(R) Core(TM) i5-13500H

	Базовая скорость:	2,60 ГГц
	Сокетов:	1
	Ядра:	12
	Логических процессоров:	16
	Виртуализация:	Включено
	Кэш L1:	1,1 МБ
	Кэш L2:	9,0 МБ
	Кэш L3:	18,0 МБ
 
GPU:

	NVIDIA GeForce RTX 4060 Laptop GPU

	Версия драйвера:	32.0.15.6636
	Дата разработки:	03.12.2024
	Версия DirectX:	12 (FL 12.1)
	Физическое расположение:	PCI-шина 1, устройство 0, функция 0

	Использование	0%
	Выделенная память графического процессора	0,0/8,0 ГБ
	Общая память графического процессора	0,0/7,9 ГБ
	Оперативная память графического процессора	0,0/15,9 ГБ




