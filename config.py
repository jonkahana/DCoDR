base_config = dict(

	save_every=20,
	eval_every=5,

	train=dict(

		learning_rate=dict(
			generator=3e-4,
			latent=3e-3,
			encoder=1e-4,
		)
	)
)
