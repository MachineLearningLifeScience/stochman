import pytest
import torch

from stochman import curves


@pytest.mark.parametrize("curve_class", [curves.DiscreteCurve, curves.CubicSpline])
class TestCurves:
    @pytest.mark.parametrize("requires_grad", [True, False])
    @pytest.mark.parametrize("batch_dim", [1, 5])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_curve_evaluation(self, curve_class, requires_grad, batch_dim, device):
        if not torch.cuda.is_available() and device == "cuda:0":
            pytest.skip("test requires cuda")

        dim = 2
        begin = torch.randn(batch_dim, dim)
        end = torch.randn(batch_dim, dim)
        num_nodes = 20
        c = curve_class(begin, end, num_nodes, requires_grad=requires_grad).to(device)
        assert isinstance(c, curves.BasicCurve)

        # assert that everything was moved to the correct device
        assert c.device == torch.device(device)
        assert c.begin.device == torch.device(device)
        assert c.end.device == torch.device(device)
        assert c.params.device == torch.device(device)

        eval_nodes = 10
        t = torch.linspace(0, 1, eval_nodes).to(device)
        out = c(t)
        assert isinstance(out, torch.Tensor)
        assert out.device == torch.device(device)

        if batch_dim == 1:
            assert list(out.shape) == [eval_nodes, dim]
        else:
            assert list(out.shape) == [batch_dim, eval_nodes, dim]

        if requires_grad:
            # make sure that parameters is non empty in this case
            # and the only parameters present are the once we defined
            assert list(c.parameters())
            assert all(list(c.parameters())[0].flatten() == c.params.flatten())

    @pytest.mark.parametrize("dim", [2, 3])
    def test_plot_func(self, curve_class, dim):
        begin = torch.randn(1, dim)
        end = torch.randn(1, dim)
        c = curve_class(begin, end, 20)
        if dim == 2:
            # this should work
            figs = c.plot()
            assert len(figs) == 1
        elif dim == 3:
            # this should raise an error
            with pytest.raises(
                ValueError, match=r"BasicCurve.plot only supports plotting curves in 1D or 2D.*"
            ):
                c.plot()

    @pytest.mark.parametrize("batch_size", [1, 5])
    def test_fit_func(self, curve_class, batch_size):
        """test fit function"""
        c = curve_class(torch.randn(batch_size, 2), torch.randn(batch_size, 2), 20)
        loss = c.fit(torch.linspace(0, 1, 10), torch.randn(5, 10, 2))
        assert isinstance(loss, torch.Tensor)

    def test_getindex_func(self, curve_class):
        """test __getidx__ function"""
        batched_c = curve_class(torch.randn(5, 2), torch.randn(5, 2))
        for i in range(len(batched_c)):
            c = batched_c[i]
            assert isinstance(c, curve_class)
            assert list(c.begin.shape) == [1, 2]
            assert list(c.end.shape) == [1, 2]
            assert c.device == batched_c.device

    def test_setindex_func(self, curve_class):
        """test __setidx__ function"""
        batched_c = curve_class(torch.randn(5, 2), torch.randn(5, 2))
        for i in range(len(batched_c)):
            batched_c[i] = curve_class(torch.randn(1, 2), torch.randn(1, 2))
            assert batched_c[i]

    def test_to_other(self, curve_class):
        """test .tospline and .todiscrete"""
        c = curve_class(torch.randn(1, 2), torch.randn(1, 2), 20)
        if curve_class == curves.DiscreteCurve:
            new_c = c.tospline()
            assert isinstance(new_c, curves.CubicSpline)
        elif curve_class == curves.CubicSpline:
            new_c = c.todiscrete()
            assert isinstance(new_c, curves.DiscreteCurve)

    def test_euclidean_length(self, curve_class):
        begin = torch.zeros(1, 2).float()
        end = torch.ones(1, 2).float()
        c = curve_class(begin, end, 20)
        el = c.euclidean_length()
        assert torch.isclose(el, torch.tensor([2.0]).sqrt())

    def test_constant_speed(self, curve_class):
        batch_size = 5
        dim = 2
        timesteps = 50
        begin = torch.randn(batch_size, dim)
        end = torch.randn(batch_size, dim)
        c = curve_class(begin, end, 20)
        new_t, Ct, curve_length = c.constant_speed(t=torch.linspace(0, 1, timesteps))
        assert isinstance(new_t, torch.Tensor)
        assert isinstance(Ct, torch.Tensor)
        assert isinstance(curve_length, torch.Tensor)
        assert new_t.shape == (batch_size, timesteps)
        assert Ct.shape == (batch_size, timesteps, dim)
        assert curve_length.shape == (batch_size,)

    def test_plotting_in_axis(self, curve_class):
        batch_size = 5
        dim = 2
        begin = torch.randn(batch_size, dim)
        end = torch.randn(batch_size, dim)
        c = curve_class(begin, end, 20)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1)
            c.plot(ax=ax)
            plt.close(fig)
            assert True
        except Exception as e:
            assert False, e
