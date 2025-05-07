import  onnxruntime as onr
import numpy as np
import cv2

session  = onr.InferenceSession("source/model/model.onnx")

print(session)
print(onr.get_available_providers())
#######################################################################
# NodeArg(name='inputNet_IN', type='tensor(uint8)', shape=[1, 3, 416, 416]):
#     1) tensor(uint8) - значение пикселя
#     2) shape=[1 - 1 изображение (batch size), 3 - 3 канала, 416, 416 - разрешение 416 x 416]
#######################################################################
print("\nInputs")
for inp in session.get_inputs():
    print(inp)


#######################################################################
#     NodeArg(name='dets', type='tensor(float)', shape=[None, 5])
#         1) Передает данные точек в виде float значений
#         2) Количество объектов None (неизвестно)
#         3) Передает 5 размерностей в одном объекте (предположительно x_max, x_min, y_max, y_min, confidence)
#
#     NodeArg(name='labels', type='tensor(int64)', shape=[None])
#         1) По идеи будет 1 класс
#######################################################################
print("\nOutputs")
for out in session.get_outputs():
    print(out)


cap = cv2.VideoCapture("rtsp://172.32.0.93/live/0")
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертация в RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ресайз до 416 x 416
    image_resize = cv2.resize(image, (416, 416))

    #######################################################################
    # Передача тензору данных происходит в след. виде: image : Channel, Height, Width, а у нас image : Width, Height, Channel.
    # Поэтому меняем оси так:
    # Новая 0-я ось ← старая 2-я (каналы),
    # Новая 1-я ось ← старая 0-я (высота),
    # Новая 2-я ось ← старая 1-я (ширина).
    #######################################################################
    tensor = np.transpose(image_resize, (2, 0, 1))
    input_tensor = tensor[np.newaxis, :] # ":" означает «все элементы» по остальным измерениям

    output_tensor = session.run(
        None, # Возврат всех входов модели
        { session.get_inputs()[0].name: input_tensor } # Передаем не вход тензор (словарь)
    )

    # Получаем выходные тензоры
    dets, labels = output_tensor

    # Масштаб
    h0, w0 = frame.shape[:2]
    scale_x = w0 / 416
    scale_y = h0 / 416


    for det, cls in zip(dets, labels):
        x_min, y_min, x_max, y_max, conf = det

        if conf > 0.6:
            # Масштабирование
            x1 = int(x_min * scale_x)
            y1 = int(y_min * scale_y)
            x2 = int(x_max * scale_x)
            y2 = int(y_max * scale_y)

            # рисуем прямоугольник и подпись
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"confidence: {conf:.2f}"
            cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()