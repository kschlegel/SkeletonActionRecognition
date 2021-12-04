import math

import pytest
import torch
import signatory

from shar.signatures._signaturelayers import (_SegmentSignatures,
                                              _SegmentSignaturesSequential)


class TestSegmentSignatures:
    @pytest.fixture
    def sequence(self, batch_size=2, channels=3, length=18):
        return torch.rand((batch_size, channels, length))

    @pytest.fixture(params=["signature", "logsignature"])
    def signature(self, request):
        if request.param == "signature":
            return {"logsignature": False}, signatory.signature
        else:
            return {"logsignature": True}, signatory.logsignature

    @pytest.fixture(params=[False, True], ids=["no startpoint", "+startpoint"])
    def include_startpoint(self, request):
        return request.param

    @pytest.fixture(params=[False, True], ids=["segment_len", "num_segments"])
    def setting(self, sequence, request):
        if request.param:
            return "num_segments"
        else:
            return "segment_len"

    @pytest.fixture(params=[4, 6])
    def segment_settings(self, sequence, setting, request):
        if setting == "segment_len":
            segment_len = request.param
            num_segments = int(math.ceil(sequence.shape[2] / segment_len))
        else:
            num_segments = request.param
            segment_len = int(math.ceil(sequence.shape[2] / num_segments))
        args = {setting: request.param}
        return args, num_segments, segment_len

    @pytest.fixture
    def segment_signature_module(self, signature, segment_settings,
                                 include_startpoint):
        return _SegmentSignatures(in_channels=3,
                                  signature_lvl=4,
                                  include_startpoint=include_startpoint,
                                  **signature[0],
                                  **segment_settings[0])

    @pytest.fixture
    def segment_signature_sequential_module(self, signature, segment_settings,
                                            include_startpoint):
        return _SegmentSignaturesSequential(
            in_channels=3,
            signature_lvl=4,
            include_startpoint=include_startpoint,
            **signature[0],
            **segment_settings[0])

    def test_segmentsignature(self, signature, include_startpoint, sequence,
                              segment_settings, segment_signature_module):
        segment_sigs = segment_signature_module(sequence)
        __, num_segments, segment_len = segment_settings

        sequence = sequence.transpose(1, 2)
        # iterate over the segments, compute signatures individually and
        # compare to the modules output
        for i in range(num_segments):
            sig = signature[1](sequence[:, i * segment_len:(i + 1) *
                                        segment_len, :],
                               depth=4)
            if include_startpoint:
                assert torch.allclose(segment_sigs[:, :-3, i], sig)
                segment_pts = [i * segment_len for i in range(num_segments)]
                assert torch.equal(segment_sigs[:, -3:, i],
                                   sequence[:, segment_pts[i], :])
            else:
                assert torch.allclose(segment_sigs[:, :, i], sig)

    def test_segmentsignaturesequential(self, signature, include_startpoint,
                                        sequence, segment_settings,
                                        segment_signature_sequential_module):
        segment_sigs = segment_signature_sequential_module(sequence)
        __, num_segments, segment_len = segment_settings

        sequence = sequence.transpose(1, 2)
        segment_pts = [i * segment_len for i in range(num_segments + 1)]
        segment_pts[-1] = sequence.shape[1]
        # iterate over the segments, compute signatures individually and
        # compare to the modules output
        for i in range(num_segments):
            sig = signature[1](sequence[:,
                                        segment_pts[i]:segment_pts[i + 1], :],
                               depth=4)
            if include_startpoint:
                assert torch.allclose(segment_sigs[:, :-3, i], sig)
                assert torch.equal(segment_sigs[:, -3:, i],
                                   sequence[:, segment_pts[i], :])
            else:
                assert torch.allclose(segment_sigs[:, :, i], sig)
