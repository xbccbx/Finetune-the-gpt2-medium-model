import pandas as pd
import re

import pandas as pd
import re

def remove_non_ascii(text):
    """移除所有非ASCII字符"""
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def remove_emoji(text):
    """移除文本中的emoji和回车，回车替换为空格"""
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.replace('\n', ' ')
    text = text.replace('"', '')
    text = text.replace(' "', ' ')
    text = text.replace('--Donald J. Trump', '')
    text = text.replace(' Donald J. Trump', '')
    text = text.replace(': ', ' ')
    text = text.replace('...', '.')
    text = text.replace('-', ' ')
    text = text.replace('    ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('@', '')
    return text

def remove_urls(text):
    """移除文本中的网址链接"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_j(text):
    """移除文本中的网址链接"""
    url_pattern = re.compile(r'pic.twitter.com/\S+')
    return url_pattern.sub(r'', text)

# 定义一个新的函数来清理文本，并去除只包含双引号的行
# 
def remove_quotes_only_rows(text):

    # 检查文本是否只包含双引号

    if text.replace('"', '').strip() == '':

        return None  # 返回None将在后续步骤中被去除

    else:

        return text

# 载入数据集
file_path = 'dataset.csv'  # 文件路径
data = pd.read_csv(file_path)

# 选择第二列并去除emoji和网址链接
processed_data = data['content'].apply(remove_emoji).apply(remove_urls).apply(remove_non_ascii).apply(remove_j)
# processed_data = processed_data.apply(remove_emoji).apply(remove_urls)

# # 移除所有只含双引号的行
processed_data = processed_data[processed_data != '""\n']
# 移除空行
processed_data = processed_data.apply(remove_quotes_only_rows).dropna()

# 保存处理后的数据
processed_file_path = 'processed_dataset.csv'  # 保存的文件路径
processed_data.to_csv(processed_file_path, index=False)
