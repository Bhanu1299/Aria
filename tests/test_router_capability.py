import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from unittest.mock import patch
from router import route


def _route_no_groq(command):
    """Route without making a real Groq API call."""
    with patch("router._classify") as mock_classify:
        mock_classify.side_effect = AssertionError("Groq should not be called for capability")
        return route(command)


def test_what_can_you_do():
    result = _route_no_groq("what can you do")
    assert result["type"] == "capability"

def test_capabilities():
    result = _route_no_groq("what are your capabilities")
    assert result["type"] == "capability"

def test_what_are_you_capable_of():
    result = _route_no_groq("what are you capable of")
    assert result["type"] == "capability"

def test_aria_features():
    result = _route_no_groq("what features do you have")
    assert result["type"] == "capability"

def test_help_me():
    result = _route_no_groq("help me")
    assert result["type"] == "capability"
