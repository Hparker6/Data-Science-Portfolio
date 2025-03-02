import gradio as gr

def GradioUI(agent):
    def predict(message):
        return agent.run(message)

    iface = gr.Interface(
        fn=predict,
        inputs="text",
        outputs="text",
        title="AI Agent Alfred",
        description="Ask me anything!"
    )
    return iface
