import numpy as np

from simulations import configs, data, run


def test_all_simulations_can_generate_data():
    for name, cfg in configs.SIMULATIONS.items():
        rng1, rng2 = run.make_generators()
        Y, labels = data.make_latent_space(rng1, configs.MEANS, run.make_covariances(), configs.NCELLS)
        sim = run.simulate(cfg, rng1, rng2, Y, labels)
        assert sim["A"].shape == sim["B"].shape == (sum(configs.NCELLS), sum(configs.NCELLS)), name
        assert np.array_equal(sim["labels_a"], sim["P0"] @ sim["labels_b"]), name


def test_blockdiag_D_assigns_every_peak_to_one_gene():
    D = data.make_blockdiag_D(1003, 100)
    assert (D.sum(1) == 1).all()


def test_slug_is_safe_for_file_names():
    assert run.slug("8'") == "sim_8prime"
