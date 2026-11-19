"""Streamlit entry point for the cosmic-web simulation project."""

import streamlit as st


def main() -> None:
    """Display the project introduction and current implementation status."""
    st.set_page_config(page_title="Cosmic Web Simulator", page_icon="🌌")
    st.title("Cosmic Web Simulator")
    st.write(
        "Explore how matter can form large-scale cosmic structures. "
        "The simulation and data-analysis modules are being built incrementally."
    )
    st.info(
        "This starter app does not yet run a physical simulation. "
        "The numerical interfaces are outlined in the `src/` package."
    )


if __name__ == "__main__":
    main()
