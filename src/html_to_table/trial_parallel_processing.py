from concurrent.futures import ProcessPoolExecutor
import time

def mywait(n):
    # Perform a more CPU-intensive dummy calculation
    total = 0
    for i in range(10**8 * n):
        total += (i ** 2) % 7
    print(f"Processed {n}")

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    # with ProcessPoolExecutor() as executor:
    #     list(executor.map(mywait, numbers))

    for n in numbers:
        mywait(n)
    
