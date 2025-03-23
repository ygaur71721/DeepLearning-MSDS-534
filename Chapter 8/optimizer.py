# Choose optimizer
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.RMSprop()]

for opt in optimizers:
    print(f"\nTraining with optimizer: {opt._name}")

    model = build_model(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024, batch_size=64)
    model.compile(optimizer=opt, loss=loss)

    history = model.fit(dataset, epochs=1)
    final_loss = history.history['loss'][-1]
    print(f"Final loss with {opt._name}: {final_loss:.4f}")
