import numpy as np
import pytest

from scripts.analysis.verifier_signal_deep import attention_decomposition, patch_states


def test_attention_decomposition_exact_and_symmetric():
    a0, a1 = np.array([.2,.8]), np.array([.6,.4])
    v0, v1 = np.array([-2.,4.]), np.array([3.,1.])
    value, routing = attention_decomposition(a0,v0,a1,v1)
    assert value.sum()+routing.sum() == pytest.approx(a1@v1-a0@v0)
    reverse = attention_decomposition(a1,v1,a0,v0)
    np.testing.assert_allclose(reverse[0],-value)
    np.testing.assert_allclose(reverse[1],-routing)
    assert attention_decomposition(a0,v0,a0,v1)[1].sum() == 0


def test_patching_preserves_other_tokens_and_originals():
    source = np.zeros((3,2))
    donor = np.ones((3,2))
    actual = patch_states(source,donor,[2])
    assert actual[:2].sum()==0 and actual[2].sum()==2
    assert source.sum()==0
    np.testing.assert_array_equal(patch_states(source,donor,[0,1,2]),donor)
    np.testing.assert_array_equal(patch_states(source,donor,[]),source)
    with pytest.raises(ValueError):
        patch_states(source,donor[:2],[0])


def test_mean_readout_matches_training_float16_roundtrip():
    import torch

    from scripts.analysis.verifier_signal_deep import logits
    from src.harness.learners import LinearHead
    head = LinearHead(2).eval()
    state = np.array([[1.001, 2.003], [1.008, 2.014]], dtype=np.float32)
    with torch.no_grad():
        expected = head(torch.from_numpy(state.mean(0).astype(np.float16).astype(np.float32))[None]).item()
    assert logits(head,[state],rep='step_mean')[0] == pytest.approx(expected)


def test_norm_patch_changes_norm_but_preserves_direction():
    from scripts.analysis.verifier_signal_deep import patch_norms
    a=np.array([[3.,4.],[1.,0.]],dtype=np.float32)
    b=np.array([[0.,2.],[4.,3.]],dtype=np.float32)
    p=patch_norms(a,b)
    np.testing.assert_allclose(np.linalg.norm(p,axis=1),np.linalg.norm(b,axis=1))
    np.testing.assert_allclose(p/np.linalg.norm(p,axis=1)[:,None],a/np.linalg.norm(a,axis=1)[:,None])
    with pytest.raises(ValueError):
        patch_norms(a,np.zeros_like(b))
