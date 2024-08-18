from dependencies.model_factory import summary_model

def make_summary(text):
    summary = summary_model.exec(text)
    return summary