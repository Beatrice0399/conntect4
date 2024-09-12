ROWS = 6
COLS = 7
id_yellow = -1
id_red = 1

right_dia = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
             [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
             [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
             [(0, 3), (1, 4), (2, 5), (3, 6)],
             [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)],
             [(2, 0), (3, 1), (4, 2), (5, 3)]
             ]

left_dia = [[(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)],
            [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)],
            [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)],
            [(0, 3), (1, 2), (2, 1), (3, 0)],
            [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2)],
            [(2, 6), (3, 5), (4, 4), (5, 3)]
            ]

def checkWin(grid: list[list[int]]):
    for row in range(ROWS):
        if grid[row][3] == 0: continue
        winner = checkList(grid[row])
        if winner: 
            return winner
    
    for index in range(COLS):
        col = []
        for x in range(6):
            col.append(grid[x][index])
        winner = checkList(col)
        if winner: 
            return winner
    
    
    for dia_coor in left_dia:
        dia = []
        for coor in dia_coor:
            dia.append(grid[coor[0]][coor[1]])
            winner = checkList(dia)
        if winner: 
            return winner
        
    for dia_coor in right_dia:
        dia = []
        for coor in dia_coor:
            dia.append(grid[coor[0]][coor[1]])
            winner = checkList(dia)
        if winner: 
            return winner
        
    return 0

def checkList(lst:list[int]):
    valid_indexes = len(lst) - 3
    winner = 0
    for index in range(valid_indexes):
        if sum(lst[index: index+4]) == 4: winner = 1
        elif sum(lst[index: index+4]) == -4: winner = -1
    return winner  