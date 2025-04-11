from cog import BasePredictor, Input
from src.app import initialize_models, CRAGPipeline

class Predictor(BasePredictor):
    def setup(self):
        # Load mô hình và pipeline
        initialize_models()
        self.pipeline = CRAGPipeline(max_hops=3, top_k=4)

    def predict(self, query: str = Input(description="Câu hỏi hoặc triệu chứng cần phân loại")) -> str:
        # Gọi pipeline xử lý truy vấn
        return self.pipeline.hop_pipeline(query)
