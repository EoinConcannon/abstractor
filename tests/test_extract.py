import extract


def test_clean_text_rejoins_words_split_by_a_hyphen():
    assert extract.clean_text("the meth-\nod works") == "the method works"


def test_clean_text_expands_ligatures():
    assert extract.clean_text("the \ufb01eld of \ufb02ow") == "the field of flow"


def test_clean_text_collapses_whitespace():
    assert extract.clean_text("one   two\n\nthree") == "one two three"


def test_normalise_makes_running_headers_comparable():
    a = extract.normalise("IEEE Access, vol. 12, p. 4")
    b = extract.normalise("IEEE   Access, vol. 12, p. 5")
    assert a == b


def test_classify_line_left_right_and_full_width():
    width = 600
    assert extract.classify_line(50, 280, width) == "left"
    assert extract.classify_line(320, 550, width) == "right"
    assert extract.classify_line(50, 550, width) == "full"


def test_reading_order_reads_columns_then_continues_below_a_full_width_block():
    def block(text, col, y0):
        return {"text": text, "col": col, "y0": y0}

    blocks = [
        block("R1", "right", 100), block("L2", "left", 200), block("title", "full", 50),
        block("L1", "left", 100), block("R2", "right", 200),
        block("wide", "full", 300), block("L3", "left", 350),
    ]
    ordered = [b["text"] for b in extract.reading_order(blocks)]
    assert ordered == ["title", "L1", "L2", "R1", "R2", "wide", "L3"]


def test_extract_pages_on_a_generated_two_column_pdf(sample_pdf):
    pages = extract.extract_pages(str(sample_pdf))

    assert [p["page"] for p in pages] == [1, 2, 3, 4]

    for page in pages:
        assert "Journal of Testing" not in page["text"]      # running header removed
    assert "101" not in pages[0]["text"]                     # footer page number removed

    page2 = pages[1]["text"]
    assert page2.index("LEFT p2 line 5") < page2.index("RIGHT p2 line 0")   # whole left column first

    page1 = pages[0]["text"]
    assert page1.index("A full width title") < page1.index("LEFT p1 line 0")  # full-width block first
