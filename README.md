# Face-recognition-with-AI-

import cv2
import time

# Carregar o classificador em cascata pré-treinado para detecção de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Dicionário para armazenar o tempo de início de carregamento para cada rosto detectado
loading_start_times = {}
# Conjunto para armazenar rostos que já completaram o carregamento
completed_faces = set()
# Dicionário para armazenar rostos congelados e o tempo em que devem ser descongelados
frozen_faces = {}
loading_duration = 2  # Duração do carregamento em segundos
reset_delay = 2  # Tempo para reiniciar o processo após a mensagem
reset_time = None  # Hora em que o processo será reiniciado
reset_in_progress = False  # Indicador de reinício em progresso

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convertendo a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção de rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()  # Tempo atual

    if reset_in_progress:
        # Verificar se o tempo de reinício foi alcançado
        if current_time - reset_time >= reset_delay:
            # Reiniciar o processo
            loading_start_times.clear()
            completed_faces.clear()
            frozen_faces.clear()
            reset_time = None
            reset_in_progress = False
    else:
        # Identificar rostos que devem ser congelados
        all_faces_frozen = True
        for (x, y, w, h) in faces:
            face_id = (x, y, w, h)

            if face_id not in loading_start_times:
                loading_start_times[face_id] = current_time

            # Calcular o tempo decorrido desde o início do carregamento
            elapsed_time = current_time - loading_start_times[face_id]
            if elapsed_time < loading_duration:
                all_faces_frozen = False  # Pelo menos um rosto ainda não está carregado

                # Desenhar o retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Desenhar a barra de carregamento
                bar_width = w
                bar_height = 20
                bar_x = x
                bar_y = y + h + 5
                progress = min(1, elapsed_time / loading_duration)  # Progresso da barra

                # Desenhar a barra de fundo
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)

                # Desenhar a barra de progresso
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            else:
                # Se o carregamento estiver completo, desenhar a mensagem "Desinfectado"
                if face_id not in completed_faces:
                    completed_faces.add(face_id)  # Adicionar ao conjunto de rostos completados
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Desinfectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Adicionar rosto ao conjunto de rostos congelados e registrar o tempo para descongelamento
                frozen_faces[face_id] = current_time + reset_delay

        # Remover rostos que não estão mais na tela
        faces_ids_on_screen = set((x, y, w, h) for (x, y, w, h) in faces)
        for face_id in list(completed_faces):
            if face_id not in faces_ids_on_screen:
                completed_faces.remove(face_id)
                if face_id in loading_start_times:
                    del loading_start_times[face_id]
                if face_id in frozen_faces:
                    del frozen_faces[face_id]

        # Verificar se todos rostos na tela estão carregados e devem ser congelados
        if all_faces_frozen:
            cv2.putText(frame, "Todas as barras carregadas", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if reset_time is None:
                # Iniciar o temporizador para reiniciar o processo
                reset_time = current_time
                reset_in_progress = True
        else:
            # Mostrar o frame com as alterações
            cv2.imshow('Face Detection with Loading Bar', frame)

    # Desenhar rostos congelados que devem ser exibidos com a mensagem
    for face_id, end_time in list(frozen_faces.items()):
        if current_time < end_time:
            (x, y, w, h) = face_id
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Desinfectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar o frame com as alterações
    cv2.imshow('Face Detection with Loading Bar', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura e feche as janelas
cap.release()
cv2.destroyAllWindows()
