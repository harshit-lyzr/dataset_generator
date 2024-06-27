import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType

st.set_page_config(
    page_title="Dataset Generator",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

api = st.sidebar.text_input("Enter Your OPENAI API KEY HERE", type="password")

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Dataset Generator💻")
st.markdown("## Welcome to the Dataset Generator!")
st.markdown(
    "This App Harnesses power of Lyzr Automata to Generate Dataset. You Need to input Your Dataset Format,fields and number of entries, this app generate Dummy Dataset.")

if api:
    openai_model = OpenAIModel(
        api_key=api,
        parameters={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    )
else:
    st.sidebar.error("Please Enter Your OPENAI API KEY")


def dataset_generation(format, fields, entries):
    dataset_agent = Agent(
        prompt_persona=f"You are a Data Engineer with over 10 years of experience.you cares about data integrity and believes in the importance of realistic datasets for meaningful analysis.",
        role="Data Engineer",
    )

    dataset = Task(
    name="Dataset generation",
    output_type=OutputType.TEXT,
    input_type=InputType.TEXT,
    model=openai_model,
    agent=dataset_agent,
    log_output=True,
    instructions=f"""
    Please generate a dataset in {format} format with the following fields:
    {fields}
    
    The dataset should contain {entries} entries.Each entry should be unique and provide a diverse representation across all fields.
    Ensure the entries are realistic and diverse.
    
    Accuracy is important, so ensure that {fields} are plausible and realistic. If using fictional data, maintain consistency and coherence within the dataset.
    Please provide the generated Dataset or output in the specified format.
    
    [!Important]Only generate Dataset nothing apart from it.
    """,
    )

    output = LinearSyncPipeline(
        name="Dataset Generation",
        completion_message="Dataset Generated!",
        tasks=[
            dataset
        ],
    ).run()
    return output[0]['task_output']


specify_format = st.selectbox("Enter format", ["CSV","Table"],placeholder="CSV")
specify_fields = st.text_area("Enter Fields", placeholder="Name: Customer Name, Age: Customer Age",height=300)
no_entries = st.number_input("Enter number of entries", placeholder="10")

if st.button("Generate"):
    solution = dataset_generation(specify_format, specify_fields, no_entries)
    st.markdown(solution)
