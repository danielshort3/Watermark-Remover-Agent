import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from watermark_remover.agent.tools import sanitize_title


def test_sanitize_title_basic():
    assert sanitize_title("At The Cross (Love Ran Red)") == "At_The_Cross_Love_Ran_Red"
    assert sanitize_title("  Amazing  Grace  ") == "Amazing_Grace"
    assert sanitize_title("A/B\\C:D*E?F\"G<H>I|J") == "A_B_C_D_E_F_G_H_I_J"

