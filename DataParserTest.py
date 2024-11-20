# Initialize manager
from KeywordContextManager import KeywordContextManager

context_manager = KeywordContextManager(
    model_name="gpt-3.5-turbo",
    max_total_tokens=3000,
    max_contexts=5
)



def create_prompt(user_input: str, system_prompt: str = "") -> str:
    # Count existing tokens
    current_tokens = context_manager.count_tokens(system_prompt + user_input)

    # Get and format contexts
    contexts = context_manager.get_relevant_contexts(
        user_input,
        current_prompt_tokens=current_tokens
    )
    formatted_contexts = context_manager.format_contexts(contexts)

    # Combine everything
    final_prompt = f"{system_prompt}\n{formatted_contexts}\n\nUser: {user_input}"

    return final_prompt

if __name__ == "__main__":
    context_manager.load_contexts('lorebooks/lorebooktest.json')
    userin = input("Context test:")

    print(create_prompt(userin))