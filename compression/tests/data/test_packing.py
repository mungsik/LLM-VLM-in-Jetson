from src.data.packing import pack_token_lists


def test_packs_and_inserts_eos_dropping_remainder():
    docs = [[1, 2, 3], [4, 5]]
    # stream = 1,2,3,eos(0),4,5,eos(0) -> blocks of 3: [1,2,3],[0,4,5]; remainder [0] dropped
    out = pack_token_lists(docs, block_size=3, eos_id=0)
    assert out == [[1, 2, 3], [0, 4, 5]]


def test_empty_input_returns_empty():
    assert pack_token_lists([], block_size=4, eos_id=0) == []


def test_all_blocks_have_exact_block_size():
    docs = [list(range(1, 21))]
    out = pack_token_lists(docs, block_size=4, eos_id=0)
    assert all(len(b) == 4 for b in out)
