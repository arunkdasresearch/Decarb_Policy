import streamlit as st
import os
from PIL import Image

# Set the title
st.title("Adam Bauer - The Timing Versus Allocation Trade-off in Politically Constrained Climate Policies")

# Acknowledgment and Research Impact
st.markdown("""
This research is based on the work of Adam Bauer at the University of Illinois at Urbana-Champaign, in collaboration with Stéphane Hallegatte and Florent McIsaac at the World Bank. Their study examines the delicate balance between optimal timing of climate action and the allocation of decarbonization efforts across economic sectors. 

For more details, visit [Adam Bauer's Website](http://ambauer.com) or view the webinar discussion on this research [here](https://e-axes.org/webinar/the-timing-versus-allocation-trade-off-in-politically-constrained-climate-policies/).

As the world navigates the urgency of climate change, policymakers must decide not only **when** to act but also **where** to focus their efforts. Poorly timed or misallocated policies can lead to wasted investments, unintended economic consequences, and missed opportunities to curb emissions effectively. This work provides a crucial framework for evaluating policy pathways, helping decision-makers strike the right balance between cost, feasibility, and long-term impact.
""")

# Sidebar Navigation: Select a Figure
st.sidebar.header("Select a Figure to View")

# Directory where figures are stored
fig_dir = "codes/figs"
figures = {
    "Decarbonization Dates": "2025-2-12-ar6_17_pfig1_decarb_dates_val.png",
    "Carbon Prices": "2025-2-12-ar6_17_pfig2_carbonprices.png",
    "End-of-Life Policy": "2025-2-12-ar6_17_pfig3_EOL.png",
    "Investment Paths": "2025-2-12-ar6_17_pfig4_investment_paths.png",
    "Sectoral Costs": "2025-2-12-ar6_17_pfig5_seccosts.png",
    "Aggregate Costs": "2025-2-12-ar6_17_pfig6_aggcosts.png"
}

# Display figure selection as buttons
selected_figure = list(figures.keys())[0]  # Default to the first figure
for figure_name in figures.keys():
    if st.sidebar.button(figure_name):
        selected_figure = figure_name

# Display the selected figure immediately (no scrolling needed)
fig_path = os.path.join(fig_dir, figures[selected_figure])

if os.path.exists(fig_path):
    image = Image.open(fig_path)
    st.image(image, caption=f"Figure: {selected_figure}", use_container_width=True, output_format="PNG")

    # Captions explaining each figure
    captions = {
        "Decarbonization Dates": "This figure illustrates projected decarbonization timelines under different policy suites.",
        "Carbon Prices": "This figure shows the evolution of carbon pricing across sectors for various policy choices.",
        "End-of-Life Policy": "This figure depicts policy impacts on the phase-out of carbon-intensive assets.",
        "Investment Paths": "Investment trends in abatement technologies across sectors for different policy scenarios.",
        "Sectoral Costs": "Comparative costs associated with emissions reductions across different industries.",
        "Aggregate Costs": "Total economic costs of each policy suite over time, highlighting nonlinear cost escalations."
    }

    st.markdown(f"### {captions[selected_figure]}")

# How we built this app (placed at the bottom)
st.markdown("""
---
### How This App Was Built  
This interactive visualization was built using **Streamlit**, a Python framework for intuitive and fast web-based data visualization. The app was developed in **Visual Studio Code**, version-controlled through **GitHub**, and deployed using **Streamlit Cloud** for public accessibility.  
""")
