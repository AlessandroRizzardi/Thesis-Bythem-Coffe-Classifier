from tensorflow import keras
import tensorflow as tf
import numpy as np

class Distiller(keras.Model):
	def __init__(self,student,teacher,temperature,alpha):
		super(Distiller,self).__init__()
		self.student = student
		self.teacher = teacher
		self.alpha =alpha
		self.temperature= temperature

	def compile(self, optimizer, metrics, student_loss, distillation_loss):
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.student_loss = student_loss
		self.distillation_loss = distillation_loss

	def train_step(self,data):
		x,y = data

		teacher_logits = self.teacher(x, training=False)

		with tf.GradientTape() as tape:
			student_logits = self.student(x, training=True)

			student_loss = self.student_loss(y, student_logits)
			distillation_loss = self.distillation_loss(
				tf.nn.softmax(teacher_logits/self.temperature, axis=1),
				tf.nn.softmax(student_logits/self.temperature, axis=1))
			
			loss = self.alpha*student_loss + (1- self.alpha)*distillation_loss*self.temperature**2

		
		gradients = tape.gradient(loss, self.student.trainable_variables)
		
		self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
		self.compiled_metrics.update_state(y, student_logits)

		results = {m.name: m.result() for m in self.metrics}
		results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
		
		return results
	
	def test_step(self, data):
		x,y = data
		predictions = self.student(x)
		student_loss = self.student_loss(y, predictions)
		self.compiled_metrics.update_state(y, predictions)

		results = {m.name: m.result() for m in self.metrics}
		results.update({"student_loss": student_loss})

		return results



class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )