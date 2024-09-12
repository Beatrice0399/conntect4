import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
from cnn import OurCNN
import connectFourAlg
from checkWinner import checkWin

transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the saved model
#model = ConvNet(num_classes=12)
model = OurCNN()
model.load_state_dict(torch.load('cnn4.model'))
model.eval()

def is_overlapping(rect1, rect2):
    # Coordinate dei rettangoli
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calcola i bordi dell'intersezione
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right > x_left and y_bottom > y_top:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2

        # Verifica se l'area di intersezione è significativa
        if intersection_area > 0.2 * min(area1, area2): 
            return True
    return False

# Apri il feed della videocamera (0 è l'indice della videocamera principale)
cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Errore: impossibile aprire la videocamera.")
    exit()

while True:
    # Cattura il frame corrente dalla videocamera
    ret, img = cap.read()
    if not ret:
        print("Errore: impossibile leggere il frame dalla videocamera.")
        break

    # Ridimensione del frame
    new_width = 500
    img_h, img_w, _ = img.shape
    scale = new_width / img_w
    img_w = int(img_w * scale)
    img_h = int(img_h * scale)
    img = cv.resize(img, (img_w, img_h), interpolation=cv.INTER_AREA)
    img_orig = img.copy()

    filtered_img = cv.bilateralFilter(img, 25, 190, 190)
    edge_detected = cv.Canny(filtered_img, 75, 150)

    contours, hierarchy = cv.findContours(edge_detected, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_list = []
    rect_list = []
    position_list = []

    # Identifica i contorni che assomigliano a cerchi
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        area = cv.contourArea(contour)
        rect = cv.boundingRect(contour)
        x_rect, y_rect, w_rect, h_rect = rect
        x_rect += w_rect / 2
        y_rect += h_rect / 2
        area_rect = w_rect * h_rect

        # Filtra i contorni che hanno un numero di vertici riconducibile a un cerchio 
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 250) & (area_rect < (img_w * img_h) / 5)) & (w_rect in range(h_rect - 10, h_rect + 10)):
            overlap = False
            for existing_rect in rect_list:
                if is_overlapping(rect, existing_rect):
                    overlap = True
                    break
        
            if not overlap:
                contours_list.append(contour)
                position_list.append((x_rect, y_rect))
                rect_list.append(rect)

    # Ritaglia i cerchi
    img_circle_contours = img_orig.copy()
    # cv.drawContours(img_circle_contours, contours_list, -1, (0, 255, 0), thickness=1)
    img_grid_overlay = img_orig.copy()
    img_grid = np.zeros([img_h,img_w,3], dtype=np.uint8)   
    
    if len(rect_list) == 0: continue

    # Cerca di capire la tabella
    rows, cols = (6,7)
    mean_w = sum([rect[2] for r in rect_list]) / len(rect_list)
    mean_h = sum([rect[3] for r in rect_list]) / len(rect_list)
    position_list.sort(key = lambda x:x[0])
    max_x = int(position_list[-1][0])
    min_x = int(position_list[0][0])
    position_list.sort(key = lambda x:x[1])
    max_y = int(position_list[-1][1])
    min_y = int(position_list[0][1])
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    col_spacing = int(grid_width / (cols-1))
    row_spacing = int(grid_height / (rows - 1))

    rect_list_circles = []

    for x_i in range(0,cols):
        x = int(min_x + x_i * col_spacing)
        for y_i in range(0,rows):
            y = int(min_y + y_i * row_spacing)
            r = int((mean_h + mean_w)/5)
            rect = (int(x - (col_spacing / 2)), int(y - (row_spacing/2)), col_spacing, row_spacing)
            rect_list_circles.append(rect)
            cv.rectangle(img_grid_overlay, rect, (255,255,0), 2)
    
    result_matrix: list = []
    
    for index, rect in enumerate(rect_list_circles):
        x, y, w, h = rect
        cropped_img = img_orig[y:y+abs(h), x:x+abs(w)]
        if len(cropped_img) == 0: continue 

        resized_img = []
    
        # Ridimensiona l'immagine ritagliata alla dimensione prevista dal modello (150, 150, 3)
        try:
            resized_img = cv.resize(cropped_img, (150, 150))           
        except:
            continue
        # Pre-processa l'immagine: normalizza e aggiungi dimensione batch
        pil_img = Image.fromarray(cv.cvtColor(resized_img, cv.COLOR_BGR2RGB))
        square_tensor = transformer(pil_img).unsqueeze(0)
        classes = ['N', 'R', 'Y']
        with torch.no_grad():
            output = model(square_tensor)
        score, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        """
        # Disegna il rettangolo e la previsione sull'immagine
        cv.rectangle(img_circle_contours, (x, y), (x + w, y + h), (0, 0, 0), 1)

        # Mostra la previsione come testo
        label_text = str(classes[predicted_class])
        
        
        if img_circle_contours is not None and x >= 0 and y - 10 >= 0:
            cv.putText(img_circle_contours, label_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        """
        result_matrix.append(predicted_class)
    
    rows, cols = (6,7)    
    id_red = 1
    id_yellow = -1
    grid = np.zeros((rows, cols))       
    
    for index, el in enumerate(result_matrix):
        row = index % 6
        col = index// 6 
        if el == 2: grid[row][col] = -1
        elif el == 1: grid[row][col] = 1
    
    winner = checkWin(grid)
    if winner: 
        try:
            font = cv.FONT_HERSHEY_COMPLEX
            if winner == 1: 
                
                text = "RED WINS"
                textsize = cv.getTextSize(text, font, 1, 2)[0]

                textX = (img_circle_contours.shape[1] - textsize[0]) / 2
                textY = (img_circle_contours.shape[0] + textsize[1]) / 2
                
                cv.putText(img_circle_contours, text, (textX, textY), font, 1, (0, 0, 255), 2)
                
            elif winner == -1:
                
                text = "YELLOW WINS"
                textsize = cv.getTextSize(text, font, 1, 2)[0]

                textX = (img_circle_contours.shape[1] - textsize[0]) // 2
                textY = (img_circle_contours.shape[0] + textsize[1]) // 2
                
                cv.putText(img_circle_contours, text, (textX, textY), font, 1, (0, 255, 255), 2)
        except: continue
        
        
    else:
        # Generate Best Move
        num_red = sum([np.count_nonzero(row == 1) for row in grid])
        num_yellow = sum([np.count_nonzero(row == -1) for row in grid])
        try:
            if not any([0 in row for row in grid]):
                grid_full = True
                
            elif num_yellow < num_red:
                move = (connectFourAlg.bestMove(grid*(-1), 1, -1), id_yellow)
            else:
                move = (connectFourAlg.bestMove(grid, 1, -1), id_red)
                

            # Display Output
            if any([0 in row for row in grid]):
                y = 0
                for y_i in range(6):
                    if grid[y_i][move[0]] == 0: y = y_i
                index = (move[0] * 6) + y
                if move[1] == id_red:
                    cv.rectangle(img_circle_contours, rect_list_circles[index], (0,0,255), thickness=5)
                if move[1] == id_yellow:
                    cv.rectangle(img_circle_contours, rect_list_circles[index], (0,255,255),thickness=5)  
        except: continue
    cv.imshow('Videocamera con Cerchi Rilevati', img_circle_contours)
        
    # Premere 'q' per uscire dal loop
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

    

# Rilascia la videocamera e chiudi tutte le finestre
cap.release()
cv.destroyAllWindows()
