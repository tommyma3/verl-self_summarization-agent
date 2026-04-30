import json

from self_summarization_agent.export import build_run_record
from self_summarization_agent.runtime import build_smoke_result


def main() -> None:
    print(json.dumps(build_run_record(build_smoke_result()), ensure_ascii=False))


if __name__ == "__main__":
    main()
