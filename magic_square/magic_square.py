import numpy as np

size = 20
magic_number = size ** 2 * (size ** 2 + 1) / size / 2
initial_state = np.arange(1, size ** 2 + 1).reshape((size, size))

def check_violations(state):
    diagonal_violations = [abs(state.trace() - magic_number), abs(state[::-1].trace() - magic_number)]
    horizontal_violations = [abs(state[i, :].sum() - magic_number) for i in range(size)]
    vertical_violations = [abs(state[:, i].sum() - magic_number) for i in range(size)]
    return horizontal_violations + vertical_violations + diagonal_violations

def swap(state, first, second):
    tmp = state.copy()[first]
    state[first] = state[second]
    state[second] = tmp
    
def step(state, violations=None, mode=1):
    if violations is None:
        violations = check_violations(state)
    most_violated = np.argmax(violations)
    sum_violations = sum(violations)
    print(sum_violations)
    # horizontal
    if most_violated < size : 
        first_place = (most_violated, np.random.randint(0, size))
    elif most_violated < 2 * size :
        first_place = (np.random.randint(0, size), most_violated - size)
    elif most_violated == 2 * size:
        index = np.random.randint(0, size)
        first_place = (index, index)
    else:
        index = np.random.randint(0, size)
        first_place = (index, size - index)
    
    if mode == 2:
        for first_place_x in range(size):
            for first_place_y in range(size):
                for second_place_x in range(size):
                    for second_place_y in range(size):
                        first_place = (first_place_x, first_place_y)
                        second_place = (second_place_x, second_place_y)
                        new_state = state.copy()
                        swap(new_state, first_place, second_place)
                        new_violations = check_violations(new_state)
                        if sum(new_violations) < sum_violations:
                            state = new_state.copy()
                            violations = new_violations
                            return state, violations
    # return state, violations
    elif mode == 1:
        while True:
            new_state = state.copy()
            first_place = (np.random.randint(0, size), np.random.randint(0, size))
            second_place = (np.random.randint(0, size), np.random.randint(0, size))
            swap(new_state, first_place, second_place)
            new_violations = check_violations(new_state)
            if sum(new_violations) < sum_violations:
                state = new_state.copy()
                violations = new_violations
                return state, violations
    raise ValueError

def run(state):
    violations = None
    k = 0
    mode = 1
    while True:
        k += 1
        state, violations = step(state, violations, mode)
        if sum(violations) == 0:
            break
        if k > 1000:
            mode = 2
        if k % 10 == 0:
            print(state)
    return state
    
    
print(initial_state)
swap(initial_state, (0,0), (size-1, size-1))
print(initial_state)

end_state = run(initial_state)
print(end_state)