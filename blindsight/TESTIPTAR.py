import pygame

# Initialize the pygame mixer
pygame.mixer.init()

# Load the sound file
sound = pygame.mixer.Sound("output.wav")

# Play the sound
sound.play()

# Wait for the sound to finish before closing the program
pygame.time.delay(int(sound.get_length() * 1000))  # Delay for the sound duration in milliseconds
