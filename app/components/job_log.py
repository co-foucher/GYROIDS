"""Renders the live status/log of an app.jobs.Job."""
import streamlit as st

from app.jobs import Job


def render_job_status(job: Job) -> None:
    """
    Shows a job's current status and log tail. Streamlit doesn't poll
    background threads on its own, so a "Refresh status" button is offered
    while the job is running - click it (or add the `streamlit-autorefresh`
    package and call it once at the top of the page) to see progress update.
    """
    if job is None:
        st.info("No job has been started yet.")
        return

    if job.status == "running":
        st.info(f"Running: {job.label}")
    elif job.status == "done":
        st.success(f"Done: {job.label}")
    else:
        st.error(f"Failed: {job.label}")
        if job.error:
            st.code(job.error)

    if job.log:
        st.code("\n".join(job.log[-200:]))

    if job.status == "running":
        if st.button("Refresh status", key=f"refresh_{job.id}"):
            st.rerun()
