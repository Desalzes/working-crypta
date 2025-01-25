import asyncio
import logging
from src.cli import CLI
import os
import torch
import nest_asyncio
import tracemalloc

tracemalloc.start()
nest_asyncio.apply()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

async def main():
    setup_logging()
    cli = CLI()
    await cli.run()

if __name__ == "__main__":
    print("MAKE SURE VPN IS ON")
    print("MAKE SURE VPN IS ON")
    print("MAKE SURE VPN IS ON")
    print("Sometimes more efficient to delete all of something and start from scratch")
    print("The problem is indicators not in csv rows, get binance_data to add them at the end")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")