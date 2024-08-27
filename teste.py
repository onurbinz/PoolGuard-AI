import cv2
import time
import numpy as np


colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (155, 0, 0)]

area_color_alpha = (0, 255, 0)  #o último parametro define a opacidade, quanto menor mais transparente

#funcao para pegar o centroid
def getCentroid(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return(cx, cy)

#funcao para desenhar o poligono
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

#pontos do poligono
pts = []

#chamada da funcao poligono
cv2.namedWindow('detections')
cv2.setMouseCallback('detections', draw_polygon)


#carrega as classes
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#captura da camera ou video
cap = cv2.VideoCapture("video_exp.mp4")

#carregando os pessos da rede neural
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#setando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

#lendo os frames do video
while True:

    #captura do frame
    ok, frame = cap.read()

    #desnhar a area
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        #colorir o interior da area com transparencia
        #overlay = frame.copy()
        #cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], area_color_alpha)
        #cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    #começo da contagem dos ms
    start = time.time()

    #detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    #fim da contagem dos ms
    end = time.time()

    #percorrer todas as detecçoes
    for (classid, score, box) in zip(classes, scores, boxes):

        #rodar apenas em pessoas
        if classid == 0:

            # Calcule o centróide
            x_center = int(box[0] + box[2]/2)
            y_center = int(box[1] + box[3]/2)

            #printar o centroid
            cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)  # Desenha um círculo verde

            #verificar se há pontos na área desenhada
            if len(pts) > 0:

                #converter pontos para tipo de dados correto (CV_32S)
                pts_int32 = np.array(pts, dtype=np.int32)

                #verificar se o centróide está dentro da área desenhada
                is_inside = cv2.pointPolygonTest(pts_int32, (x_center, y_center), False)

                if is_inside > 0:
                    #pessoa está dentro da área
                    pessoa_dentro = True
                else:
                    #pessoa está fora da área
                    pessoa_dentro = False

            else: #caso os pontos sejam limpos continuar rodando
                pessoa_dentro = False

            print(pessoa_dentro)

            #gerando uma cor pra classe
            color = colors[int(classid) % len(colors)]

            #pegando o nome da classe pelo id e o seu score de acuracia
            score *= 100
            label = f"{class_names[classid]} : {score:.2f}%"

            #desenhando a box de detecção
            cv2.rectangle(frame, box, color, 2)

            #escrevendo o nome da classe em cima do box do objeto
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #calculando o tempo quie levou para fazer a detecção
    fps_label = f"FPS : {round((1.0/(end - start)), 2)}"

    #escrevendo o fps na tela
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    #mostrando a imagem
    cv2.imshow("detections", frame)

    #fechar o video
    if cv2.waitKey(1) == ord('q'):
        break

#liberacao da camera e destoi todas as janelas
cap.release()
cv2.destroyAllWindows()