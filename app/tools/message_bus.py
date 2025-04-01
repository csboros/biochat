"""Message bus system for handling status updates and notifications across the application.

This module provides a pub/sub pattern implementation using Streamlit's session state
to manage status updates and notifications between different components of the application.
"""

from typing import Callable
import streamlit as st


# pylint: disable=no-member
class MessageBus:
    """A simple message bus for handling status updates across the application."""

    def __init__(self):
        # Initialize subscribers in session state if it doesn't exist
        if "subscribers" not in st.session_state:
            st.session_state.subscribers = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if "subscribers" not in st.session_state:
            st.session_state.subscribers = {}
        if event_type not in st.session_state.subscribers:
            st.session_state.subscribers[event_type] = []
        st.session_state.subscribers[event_type].append(callback)

    def publish(self, event_type: str, data=None):
        """Publish an event with optional data."""
        if "subscribers" not in st.session_state:
            st.session_state.subscribers = {}
        if event_type in st.session_state.subscribers:
            for callback in st.session_state.subscribers[event_type]:
                callback(data)

message_bus = MessageBus()
