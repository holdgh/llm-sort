import click
import llm
import sys
from functools import cmp_to_key

# This default prompt template is used to ask the LLM which of two lines is more relevant
# given a query. It expects the answer to start with "Passage A" or "Passage B".
DEFAULT_PAIRWISE_PROMPT = """
Given the query:
{query}

Compare the following two lines:

Line A:
{docA}

Line B:
{docB}

Which line is more relevant to the query? Please answer with "Line A" or "Line B".
""".strip()


@llm.hookimpl  # 用以标记当前函数为llm的钩子函数
def register_commands(cli):
    @cli.command(context_settings=dict(ignore_unknown_options=True))
    @click.option(
        "--query",
        required=True,
        help="The query to use for semantic sorting. Lines will be sorted based on this query."
    )
    @click.option(
        "--method",
        type=click.Choice(["allpair", "sorting", "sliding"]),
        default="sorting",
        help="Semantic sorting method to use:\n"
             "  allpair  - Compare every pair and aggregate scores.\n"
             "  sorting  - Use a sorting algorithm with pairwise comparisons.\n"
             "  sliding  - Use a sliding-window (bubble sort) approach."
    )
    @click.option(
        "--top-k",
        type=int,
        default=0,
        help="Only keep the top K sorted lines (0 to keep all)."
    )
    @click.option("-m", "--model", help="LLM model to use for semantic sorting.")
    @click.option("--prompt", help="Custom pairwise ranking prompt template.")
    @click.argument("files", type=click.File("r"), nargs=-1)
    def sort(query, method, top_k, model, prompt, files):
        """
        Sort input lines semantically

        This command reads lines either from the FILES provided as arguments or, if no files
        are given, from standard input. Each line is treated as a separate document. The lines are then
        sorted semantically using an LLM with one of three pairwise ranking methods:

          • allpair  — PRP-Allpair: Compare every pair of lines and aggregate scores.
          • sorting  — PRP-Sorting: Use pairwise comparision with a sorting algorithm.
          • sliding  — PRP-Sliding-K: Perform a sliding-window (bubble sort) pass repeatedly.

        Example usage:
            llm sort --query "Which name is more suitable for a pet monkey?" names.txt

        The sorted lines are written to standard output.
        """
        # If no files are provided, default to reading from standard input.
        if not files:  # 非文件读取时，从标准输入中读取
            files = [sys.stdin]

        documents = []
        for f in files:
            for line in f:
                # Remove the trailing newline (but preserve other whitespace)
                line = line.rstrip("\n")  # 去除行尾的换行符
                # Only add non-empty lines.
                if line:  # 非空行作为独立的文档信息【id和文本内容】
                    documents.append({"id": str(len(documents)), "content": line})

        if not documents:
            click.echo("No input lines provided.", err=True)
            return

        # Initialize the LLM model.
        from llm.cli import get_default_model
        from llm import get_key
        model_obj = llm.get_model(model or get_default_model())
        if model_obj.needs_key:
            model_obj.key = get_key("", model_obj.needs_key, model_obj.key_env_var)

        # Use the custom prompt if provided; otherwise, use the default.
        prompt_template = prompt or DEFAULT_PAIRWISE_PROMPT

        # Define a helper function that compares two lines (documents) by calling the LLM twice
        # (swapping the order to mitigate bias) and returning:
        #   1  => First line is preferred.
        #  -1  => Second line is preferred.
        #   0  => Tie or inconclusive.
        def pairwise_decision(query, docA, docB):  # 论文中的Pairwise Ranking Prompting (PRP)算法实现
            # First prompt: (docA, docB)
            prompt1 = prompt_template.format(query=query, docA=docA, docB=docB)
            response1 = model_obj.prompt(prompt1, system="You are a helpful assistant.").text().strip()

            # Second prompt: (docB, docA)
            prompt2 = prompt_template.format(query=query, docA=docB, docB=docA)
            response2 = model_obj.prompt(prompt2, system="You are a helpful assistant.").text().strip()

            # Normalize responses.
            resp1 = response1.lower()
            resp2 = response2.lower()
            # 对于颠倒顺序的两个提示词【去除顺序的影响】而言，如果大模型的输出结果一致，则确定相对顺序。反之平局。
            if resp1.startswith("line a") and resp2.startswith("line b"):  # docA无论在第一个还是在第二个位置，模型认为其都是与query更相关的。
                return 1   # docA is preferred
            elif resp1.startswith("line b") and resp2.startswith("line a"):  # docB无论在第一个还是在第二个位置，模型认为其都是与query更相关的。
                return -1  # docB is preferred
            else:
                return 0   # Tie or inconclusive

        # Sort the documents using the selected method.
        sorted_docs = []
        if method == "allpair":
            # PRP-Allpair: Compare every pair and aggregate scores.
            # 初始化文档得分为0
            for doc in documents:
                doc["score"] = 0.0
            n = len(documents)
            # 将文档两两配对，组合情况n(n-1)种
            for i in range(n):
                for j in range(i + 1, n):
                    decision = pairwise_decision(query, documents[i]["content"], documents[j]["content"])
                    if decision == 1:
                        documents[i]["score"] += 1.0
                    elif decision == -1:
                        documents[j]["score"] += 1.0
                    else:
                        documents[i]["score"] += 0.5
                        documents[j]["score"] += 0.5
            sorted_docs = sorted(documents, key=lambda d: d["score"], reverse=True)  # 按照文档的得分从大到小排序

        elif method == "sorting":
            # PRP-Sorting: Use a custom comparator with a sorting algorithm.
            def compare_docs(a, b):  # 比较两个文档对于query的相关性大小，作为两个文档的相对得分大小，就像两个数的比较大小一样
                decision = pairwise_decision(query, a["content"], b["content"])
                if decision == 1:
                    return -1  # a should come before b
                elif decision == -1:
                    return 1   # b should come before a
                else:
                    return 0
            sorted_docs = sorted(documents, key=cmp_to_key(compare_docs))  # 直接按照文档间的相对相关性大小排序【论文中采用堆排序，作用一样，只是方式不同】

        elif method == "sliding":  # 类似于冒泡排序，不过这里是把最大值放到最前面，最终得到相关性从大到小的文档列表
            # PRP-Sliding-K: Perform K sliding-window passes (similar to bubble sort).
            sorted_docs = documents[:]
            n = len(sorted_docs)
            for _ in range(top_k or n):  # range(top_k or n)表示--当top_k为0时，相当于range(n)；当top_k非0时，相当于range(top_k)。这里执行top_k次“冒泡”，是考虑了大模型响应的随机性，多次执行利于稳定。
                # Traverse from right to left.
                for i in reversed(range(n - 1)):  # i遍历n-2,n-3,n-4,...,0
                    decision = pairwise_decision(query, sorted_docs[i]["content"], sorted_docs[i + 1]["content"])
                    if decision == -1:  # 如果后者【例如：n-2与n-1，这里的后者指n-1】与问题更相关，则互换位置。相当于冒泡中寻找最大值，只不过这里将最大值放到了前面【冒泡是放到后面】
                        sorted_docs[i], sorted_docs[i + 1] = sorted_docs[i + 1], sorted_docs[i]
        else:
            click.echo("Invalid sorting method specified.", err=True)
            return

        if top_k > 0:
            sorted_docs = sorted_docs[:top_k]

        # Output the sorted lines to standard output.
        for doc in sorted_docs:
            click.echo(doc["content"])
