
seq_templates = [
    'Given the following purchase history of user_{}: item_{}, predict next possible item to be purchased by the user.',
    'I find the purchase history list of user_{}: item_{}. I wonder what is the next item to recommend to the user. Can you help me decide?',
    'Here is the purchase history list of user_{}: item_{}. Try to recommend next item to the user.',
    'Given the following purchase history of user_{}: item_{}, predict next possible item for the user.',
    'Based on the purchase history of user_{}: item_{}, can you decide the next item likely to be purchased by the user?',
    'Here is the purchase history of user_{}: item_{}. What to recommend next for the user?',
    'According to the purchase history of user_{}: item_{}, can you recommend the next possible item to the user?',
    'user_{} item_{}',
]

topn_templates = [
    'Which item of the following to recommend for user_{}? item_{}',
    'Choose the best item from the candidates to recommend for user_{}? item_{}',
    'Pick the most suitable item from the following list and recommend to user_{}: item_{}',
    'We want to make recommendation for user_{}. Select the best item from these candidates: item_{}',
    'user_{} item_{}',
]

exp_templates = [
    'Generate an explanation for user_{} about this product: item_{}',
    'Can you help generate an explanation of user_{} for item_{}?',
    'Help user_{} generate an explanation about this product: item_{}',
    'Generate user_{}\'s purchase explanation about item_{}',
    'Help user_{} generate an explanation for item_{}',
    'Can you help generate an explanation for user_{} about the product: item_{}',
    'Write an explanation for user_{} about item_{}',
    'Generate an explanation for user_{} about item_{}',
    'user_{} item_{}',
]


rea_templates = [
    'Generate user_{}\'s preference',
    'Generate item_{}\'s attributions',
]

rev_templates = [
    # 直接指令型
    'Generate a Review Summary for user_{} about item_{}.',
    'Predict the Review Summary of user_{} for item_{}.',
    'Output the Review Summary for user_{} concerning item_{}.',
    'Write user_{}\'s Review Summary for product item_{}.',

    # 疑问型/帮助型 (仿照您的例子)
    'What is the Review Summary for user_{} and item_{}?',
    'Can you provide the Review Summary for user_{} regarding item_{}?',
    'Help generate a Review Summary from user_{} for item_{}.', # 与您的风格类似
    'What Review Summary would user_{} likely give for item_{}?',

    # 强调“预测”
    'Predict user_{}\'s Review Summary about item_{}.',
    'Generate the predicted Review Summary for the interaction between user_{} and item_{}.',

    # 简洁/符号化 (仿照您最简洁的例子，但指明任务)
    'user_{} item_{} -> Review Summary', # 符号化，表示转换任务
    'Review Summary: user_{}, item_{}.'  # 简洁，但明确任务
]

# 使用示例 (假设您有一个填充函数)
# user_id = "123"
# item_id = "456"
# chosen_template = review_summary_prediction_templates_for_t5[0]
# filled_prompt = chosen_template.format(user_id, item_id) # 如果您的格式化是这样
# 或者更常见的是直接替换占位符
# filled_prompt = chosen_template.replace("user_{}", f"user_{user_id}").replace("item_{}", f"item_{item_id}")
# print(filled_prompt)
# 输出:
# Generate a Review Summary for user_123 about item_456.
