import json
import random
import os

class ShareGPTLoader:
    def __init__(self, file_path="sharegpt/common_en_70k.jsonl"):
        """
        初始化ShareGPT数据加载器
        
        Args:
            file_path (str): common_en_70k.jsonl文件的路径
        """
        self.file_path = file_path
        self.all_qa_pairs = []
        self.loaded = False
        
    def load_data(self):
        """
        将common_en_70k.jsonl文件中的所有对话加载到内存中
        """
        if not os.path.exists(self.file_path):
            print(f"错误: 文件 {self.file_path} 不存在")
            return False
            
        try:
            print(f"正在从 {self.file_path} 加载数据...")
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # 提取对话中的所有QA对
                        if "conversation" in data:
                            for conv in data["conversation"]:
                                if "human" in conv and "assistant" in conv:
                                    self.all_qa_pairs.append({
                                        "human": conv["human"],
                                        "assistant": conv["assistant"],
                                        "conversation_id": data.get("conversation_id", f"unknown_{line_num}"),
                                        "category": data.get("category", "unknown"),
                                        "source_line": line_num
                                    })
                    
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num} 行JSON解析错误: {e}")
                        continue
                        
            self.loaded = True
            print(f"数据加载完成! 总共加载了 {len(self.all_qa_pairs)} 个问答对")
            return True
            
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return False
    
    def get_random_qa(self):
        """
        随机抽取一个QA对
        
        Returns:
            dict: 包含human和assistant的对话内容，如果未加载数据或数据为空则返回None
        """
        if not self.loaded:
            print("错误: 数据尚未加载，请先调用load_data()")
            return None
            
        if not self.all_qa_pairs:
            print("错误: 没有可用的QA对")
            return None
            
        return random.choice(self.all_qa_pairs)
    
    def get_qa_by_index(self, index):
        """
        根据索引获取QA对
        
        Args:
            index (int): QA对的索引
            
        Returns:
            dict: 包含human和assistant的对话内容，如果索引越界则返回None
        """
        if not self.loaded:
            print("错误: 数据尚未加载，请先调用load_data()")
            return None
            
        if index < 0 or index >= len(self.all_qa_pairs):
            print(f"错误: 索引 {index} 超出范围 [0, {len(self.all_qa_pairs)-1}]")
            return None
            
        return self.all_qa_pairs[index]
    
    def get_random_qa_simple(self):
        """
        随机抽取一个QA对，只返回简化的human和assistant内容
        
        Returns:
            dict: 只包含human和assistant字段的字典
        """
        qa_pair = self.get_random_qa()
        if qa_pair:
            return {
                "human": qa_pair["human"],
                "assistant": qa_pair["assistant"]
            }
        return None
    
    def get_multiple_random_qa(self, count=5):
        """
        随机抽取多个QA对
        
        Args:
            count (int): 要抽取的QA对数量
            
        Returns:
            list: 包含多个QA对的列表
        """
        if not self.loaded or not self.all_qa_pairs:
            return []
            
        count = min(count, len(self.all_qa_pairs))
        return random.sample(self.all_qa_pairs, count)
    
    def get_stats(self):
        """
        获取数据统计信息
        
        Returns:
            dict: 包含统计信息的字典
        """
        if not self.loaded:
            return {"status": "数据未加载"}
            
        return {
            "status": "数据已加载",
            "qa_pairs_count": len(self.all_qa_pairs),
            "file_path": self.file_path
        }


# 使用示例
def main():
    # 初始化加载器 - 替换为你的实际文件路径
    loader = ShareGPTLoader("common_en_70k.jsonl")
    
    # 加载数据
    if not loader.load_data():
        return
    
    # 获取统计信息
    stats = loader.get_stats()
    print(f"数据统计: {stats}")
    
    # 示例1: 随机抽取一个完整的QA对
    print("\n" + "="*70)
    print("示例1: 随机抽取一个完整的QA对")
    print("="*70)
    qa_pair = loader.get_random_qa()
    if qa_pair:
        print(f"对话ID: {qa_pair['conversation_id']}")
        print(f"分类: {qa_pair['category']}")
        print(f"来源行: {qa_pair['source_line']}")
        print("-" * 70)
        print(f"Human: {qa_pair['human']}")
        print("-" * 70)
        print(f"Assistant: {qa_pair['assistant']}")
    
    # 示例2: 按索引获取QA对
    print("\n" + "="*70)
    print("示例2: 按索引获取QA对")
    print("="*70)
    qa_pair = loader.get_qa_by_index(0)
    if qa_pair:
        print(f"Human: {qa_pair['human']}")
        print("-" * 70)
        print(f"Assistant: {qa_pair['assistant']}")
    
    # 示例3: 随机抽取多个QA对
    print("\n" + "="*70)
    print("示例3: 随机抽取多个QA对")
    print("="*70)
    multiple_qa = loader.get_multiple_random_qa(3)
    for i, qa in enumerate(multiple_qa, 1):
        print(f"QA对 #{i}:")
        print(f"  Human: {qa['human'][:100]}...")  # 只显示前100个字符
        print(f"  Assistant: {qa['assistant'][:100]}...")
        print()


# 独立使用函数
def initialize_sharegpt_loader(file_path="common_en_70k.jsonl"):
    """
    初始化并加载ShareGPT数据的便捷函数
    
    Args:
        file_path (str): common_en_70k.jsonl文件的路径
        
    Returns:
        ShareGPTLoader: 初始化并加载完成的加载器实例
    """
    loader = ShareGPTLoader(file_path)
    if loader.load_data():
        return loader
    else:
        return None

def get_random_qa_from_common_en_70k(loader):
    """
    从已加载的ShareGPT数据中随机抽取一个QA对
    
    Args:
        loader (ShareGPTLoader): 已初始化的加载器实例
        
    Returns:
        dict: 随机QA对
    """
    if loader and isinstance(loader, ShareGPTLoader):
        return loader.get_random_qa()
    return None