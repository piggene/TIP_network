import argparse
from agent import Agent

def main():
    parser = argparse.ArgumentParser(description='mode 0 : Learning stage 1 / mode 1 : Learning stage 2')

    # Add arguments for mode 
    parser.add_argument('--mode', type=int, default=0, help='Learning mode')
    args = parser.parse_args()

    # Access arguments using args.mode
    agent = Agent(args.mode)
    agent.train()

if __name__ == '__main__':
    main()