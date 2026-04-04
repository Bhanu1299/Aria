"""Tests for _parse_salary_filter() and _parse_filters() salary branch."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jobs import _parse_salary_filter, _parse_filters


def test_salary_100k():
    assert _parse_salary_filter("find SWE jobs paying 100k") == "4"

def test_salary_120k():
    assert _parse_salary_filter("find jobs 120k or more") == "5"

def test_salary_140k():
    assert _parse_salary_filter("roles paying 140k") == "6"

def test_salary_80k():
    assert _parse_salary_filter("jobs paying around 80k") == "3"

def test_salary_six_figures():
    assert _parse_salary_filter("six figure jobs") == "4"

def test_salary_six_figures_hyphenated():
    assert _parse_salary_filter("six-figure job") == "4"

def test_salary_no_mention():
    assert _parse_salary_filter("find me remote SWE jobs") == ""

def test_parse_filters_includes_salary():
    filters = _parse_filters("find remote SWE jobs paying 120k")
    assert filters.get("f_WT") == "2"    # remote
    assert filters.get("f_SB2") == "5"   # 120k+

def test_parse_filters_no_salary():
    filters = _parse_filters("find remote SWE jobs")
    assert "f_SB2" not in filters
