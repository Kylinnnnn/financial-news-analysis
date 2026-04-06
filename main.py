from __future__ import annotations

from rss_news_fetcher import run_local_file_analysis, run_rss_analysis


MENU_ACTIONS = {
    "1": ("RSS News Analysis", run_rss_analysis),
    "2": ("Local File Analysis", run_local_file_analysis),
    "3": ("Exit Program", None),
}


def main() -> None:
    while True:
        print("\n==============================")
        print("    Financial News Sentiment Analysis Main Program    ")
        print("==============================")
        print("1. Fetch RSS financial news and perform sentiment analysis")
        print("2. Analyze local CSV format news data")
        print("3. Exit program")
        print("==============================")

        choice = input("Please select an operation option (1/2/3): ").strip()
        action = MENU_ACTIONS.get(choice)

        if action is None:
            print("[Warning] Invalid input option, please reselect.")
            continue

        action_name, action_func = action

        if choice == "3":
            print("Program exited normally.")
            break

        try:
            assert action_func is not None
            action_func()
        except Exception as e:
            print(f"[Error] {action_name} execution failed: {e}")


if __name__ == "__main__":
    main()