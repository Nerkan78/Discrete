import numpy as np




class Queen:
    def __init__(self, position, board, idx):
        self.position = position
        self.board = board
        self.violation = 0
        self.idx = idx
        self.crossed = []
    def check_right(self, change_other = True):
        # go right-up
        i, j = self.position
        while i > 0 and j < self.board.size-1:
            i -= 1
            j += 1
            is_queen, other = self.board.is_taken(i, j)
            if is_queen:
                self.violation += 1
                self.crossed.append(other.idx)
                if change_other : 
                    other.violation += 1
                    other.crossed.append(self.idx)
        # go right
        i, j = self.position
        while j < self.board.size-1:
            j += 1
            is_queen, other = self.board.is_taken(i, j)
            if is_queen:
                self.violation += 1
                self.crossed.append(other.idx)
                if change_other : 
                    other.violation += 1
                    other.crossed.append(self.idx)
        # go right-down
        i, j = self.position
        while j < self.board.size-1 and j < self.board.size-1:
            i += 1
            j += 1
            is_queen, other = self.board.is_taken(i, j)
            if is_queen:
                self.violation += 1
                self.crossed.append(other.idx)
                if change_other : 
                    other.violation += 1
                    other.crossed.append(self.idx)

        
               
    
class Board:
    def __init__(self, number_of_queens):
        self.queens = [Queen((np.random.randint(0, number_of_queens), j), self, j) for j in range(number_of_queens)]
        self.size = number_of_queens
    
    def visualise(self):
        indices = np.array([queen.position for queen in self.queens]).astype(int)

        board = np.zeros((self.size, self.size)).astype(int)
        for index in indices:
            board[tuple(index)] = 1


        print(board)
        
    def print_queens(self):
        print('\n ---------------------QUEENS---------------------')
        for queen in self.queens:
            print(f'ID {queen.idx} POSITION {queen.position} VIOLATIONS {queen.violation} CROSSED {queen.crossed}')
        self.visualise
    def is_taken(self, i ,j):
        queen = self.queens[j]
        if queen.position[0] == i:
            return True, queen
        return False, None
    
    
    def check_position(self, i, j):
        summ = i + j
        diff = i - j
        
        total_violation = 0
        crossed_queens = []
        # main diagonal 
        start_j = 0
        start_i = i - j
        while start_i < 0:
            start_i += 1
            start_j += 1
        
        while start_i < self.size and start_j < self.size:
            if start_i == i and start_j == j:
                start_i += 1
                start_j += 1
                continue
            is_queen, queen = self.is_taken(start_i, start_j)
            if is_queen:
                total_violation += 1
                crossed_queens.append(queen.idx)
            start_i += 1
            start_j += 1

        # second diagonal 
        start_j = 0
        start_i = i + j
        while start_i >= self.size:
            start_i -= 1
            start_j += 1
        
        while start_i >= 0 and start_j < self.size:
            if start_i == i and start_j == j:
                start_i -= 1
                start_j += 1
                continue
            is_queen, queen = self.is_taken(start_i, start_j)
            if is_queen:
                total_violation += 1
                crossed_queens.append(queen.idx)
            start_i -= 1
            start_j += 1
 
        # horizontal 
        start_j = 0
        start_i = i
        
        while start_j < self.size:
            if start_i == i and start_j == j:
                start_j += 1
                continue
            is_queen, queen = self.is_taken(start_i, start_j)
            if is_queen:
                total_violation += 1
                crossed_queens.append(queen.idx)
            start_j += 1
   
        return i, total_violation, crossed_queens
        
        
    
    def check_queens(self):
        for queen in self.queens:
            queen.check_right()
    def move_queen(self):
        
        violated_queen = max(self.queens, key = lambda x: x.violation)
        for queen_idx in violated_queen.crossed:
            queen = self.queens[queen_idx]
            queen.violation -= 1
            queen.crossed.remove(violated_queen.idx)
        
        position_and_queens = map(lambda i : self.check_position(i, violated_queen.idx), [i for i in range(violated_queen.position[0])] + [i for i in range(violated_queen.position[0]+1, self.size)])
        best_position, total_violation, queens = min(position_and_queens, key = lambda x : x[1])
        print(f'Previous position {violated_queen.position[0]} next position {best_position}')
        violated_queen.position = (best_position, violated_queen.idx)
        violated_queen.violation = total_violation
        violated_queen.crossed = queens
        for queen_idx in queens:
            queen = self.queens[queen_idx]
            queen.violation += 1
            queen.crossed.append(violated_queen.idx)
            
            
        
            
    def place_queens(self):
        self.check_queens()
        
        while any([queen.violation > 0 for queen in self.queens]):
            self.print_queens()
            self.visualise()
            self.move_queen()
            
            
number_of_queens = 6         
board = Board(number_of_queens)
board.visualise()
board.place_queens()

board.visualise()




























        