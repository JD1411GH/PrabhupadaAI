from trial import Trial
import sys
import os
import time

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    t = Trial()
    t.run()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_secs = end_time - start_time
    elapsed_mins = elapsed_secs / 60
    print(f"Time taken: {elapsed_mins:.2f} minutes")
