import pygame
import os
import random
import neat

ai_playing = True
generation = 0
max_score = 0

pygame.font.init()
FONT = pygame.font.SysFont('Fira Code', 16)

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 600

FRAME_RATE = 30

BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'background.png')))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'ground.png')))
CACTUS_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'small_cactus_1.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'small_cactus_2.png')))
]
DINOSAUR_IDLE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'dino.png')))
DINOSAUR_RUNNING_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'dino_running_1.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'dino_running_2.png')))
]
DINOSAUR_JUMPING_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'dino_jumping.png')))


class Dinosaur:
    IDLE_IMAGE = DINOSAUR_IDLE_IMAGE
    ANIMATION_TIME = 4
    RUNNING_IMAGES = DINOSAUR_RUNNING_IMAGES

    def __init__(self, x_axis, y_axis, ground_y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.speed = 0
        self.time = 0
        self.height = self.y_axis
        self.actual_image_count = 0
        self.actual_image = self.IDLE_IMAGE
        self.ground_y_axis = ground_y_axis

    def jump(self):
        if self.is_grounded():
            self.speed = -13.5
            self.time = 0
            self.height += self.y_axis

    def move(self):
        # calculate displacement
        self.time += 1
        displacement = 1.5 * (self.time ** 2) + self.speed * self.time

        # restrict displacement
        if displacement > 16:
            displacement = 16
        elif displacement < 0:
            displacement -= 2

        self.y_axis += displacement

    def run(self):
        self.actual_image_count += 1

        if self.actual_image_count < self.ANIMATION_TIME:
            self.actual_image = self.RUNNING_IMAGES[0]
        elif self.actual_image_count < self.ANIMATION_TIME * 2:
            self.actual_image = self.RUNNING_IMAGES[1]
        elif self.actual_image_count >= self.ANIMATION_TIME * 2 + 1:
            self.actual_image = self.RUNNING_IMAGES[0]
            self.actual_image_count = 0

    def spawn(self, screen):
        self.run()

        if self.is_grounded():
            self.y_axis = 405
        else:
            self.actual_image = self.IDLE_IMAGE

        screen.blit(self.actual_image, (self.x_axis, self.y_axis))

    def get_image_mask(self):
        return pygame.mask.from_surface(self.actual_image)

    def is_grounded(self):
        return (self.y_axis + self.actual_image.get_height()) > self.ground_y_axis


class Cactus:
    def __init__(self, x_axis, speed):
        self.x_axis = x_axis
        self.y_axis = 424
        self.IMAGE = CACTUS_IMAGES[random.choice([0, 1])]
        self.has_passed = False
        self.speed = speed

    def move(self):
        self.x_axis -= self.speed

    def spawn(self, screen):
        screen.blit(self.IMAGE, (self.x_axis, self.y_axis))

    def collide(self, dinosaur):
        dinosaur_mask = dinosaur.get_image_mask()
        cactus_mask = pygame.mask.from_surface(self.IMAGE)

        objects_distance = (self.x_axis - dinosaur.x_axis, self.y_axis - round(dinosaur.y_axis))

        has_collided = dinosaur_mask.overlap(cactus_mask, objects_distance)

        return has_collided


class Ground:
    WIDTH = GROUND_IMAGE.get_width()
    IMAGE = GROUND_IMAGE

    def __init__(self, y_axis, speed):
        self.y_axis = y_axis
        self.x_axis_0 = 0
        self.x_axis_1 = self.WIDTH
        self.speed = speed

    def move(self):
        self.x_axis_0 -= self.speed
        self.x_axis_1 -= self.speed

        if self.x_axis_0 + self.WIDTH < 0:
            self.x_axis_0 = self.x_axis_1 + self.WIDTH
        if self.x_axis_1 + self.WIDTH < 0:
            self.x_axis_1 = self.x_axis_0 + self.WIDTH

    def spawn(self, screen):
        screen.blit(self.IMAGE, (self.x_axis_0, self.y_axis))
        screen.blit(self.IMAGE, (self.x_axis_1, self.y_axis))


def render_screen(screen, ground, score, dinosaurs, cacti, speed, obstacles):
    screen.blit(BACKGROUND_IMAGE, (0, 0))

    for dinosaur in dinosaurs:
        dinosaur.spawn(screen)

    for cactus in cacti:
        cactus.spawn(screen)

    score_text = FONT.render(
        f'Velocidad: {round(speed)} | Obstáculo: {obstacles} | Puntuación Act: {round(score)} | Puntación Max: {round(max_score)}',
        True,
        (0, 0, 0)
    )
    screen.blit(score_text, (SCREEN_WIDTH - 10 - score_text.get_width(), 10))

    if ai_playing:
        score_text = FONT.render(f'Generación: {generation-1} | Dinos: {len(dinosaurs)}', True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

    ground.spawn(screen)

    pygame.display.update()


def main(genomes, config):
    global generation
    global max_score
    generation += 1

    speed = 10
    score = 0
    obstacles = 0

    ground = Ground(475, speed)
    cacti = [Cactus(SCREEN_WIDTH, speed)]
    if ai_playing:
        networks = []
        genomes_list = []
        dinosaurs = []

        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            networks.append(net)
            genome.fitness = 0
            genomes_list.append(genome)
            dinosaurs.append(Dinosaur(120, 405, ground.y_axis))
    else:
        dinosaurs = [Dinosaur(120, 405, ground.y_axis)]

    reference_values_to_spawn_obstacles = [2, 2.25, 2.5, 2.75, 3]
    reference_values_to_set_new_x_axis = [0, 0.25, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    is_running = True
    while is_running:
        clock.tick(FRAME_RATE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                pygame.quit()
                quit()

            if not ai_playing:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        for dinosaur in dinosaurs:
                            dinosaur.jump()

                    if event.key == pygame.K_r and len(dinosaurs) < 1:
                        speed = 10
                        score = 0
                        obstacles = 0
                        dinosaurs.append(Dinosaur(120, 405, ground.y_axis))
                        cacti = [Cactus(SCREEN_WIDTH, speed)]

        if ai_playing:
            cactus_index = 0
            if len(dinosaurs) > 0:
                if len(cacti) > 1 and dinosaurs[0].x_axis > (cacti[0].x_axis + cacti[0].IMAGE.get_width()):
                    cactus_index = 1
            else:
                is_running = False
                break

        for i, dinosaur in enumerate(dinosaurs):
            dinosaur.move()

            if ai_playing:
                genomes_list[i].fitness += 0.1

                # output => Entre -1 y 1  Función de Activación
                output = networks[i].activate((
                    dinosaur.y_axis,
                    abs(dinosaur.y_axis - cacti[cactus_index].IMAGE.get_height()),
                    abs(dinosaur.y_axis - cacti[cactus_index].y_axis),
                    abs(dinosaur.x_axis - cacti[cactus_index].x_axis),
                    speed
                ))

                if output[0] > 0.5:
                    dinosaur.jump()

        if len(dinosaurs):
            ground.move()

            if score > 100 and round(score) % 100 == 0:
                speed += speed * 10 / 100

            time = clock.get_time() / 1000
            score += speed * time

        has_to_add_cactus = False
        cacti_to_remove = []
        for cactus in cacti:
            has_to_add_cactus = cactus.x_axis < SCREEN_WIDTH / random.choice(reference_values_to_spawn_obstacles)

            for i, dinosaur in enumerate(dinosaurs):
                if cactus.collide(dinosaur):
                    if score > max_score:
                        max_score = score

                    dinosaurs.pop(i)

                    if ai_playing:
                        genomes_list[i].fitness -= 1
                        genomes_list.pop(i)
                        networks.pop(i)
                if not cactus.has_passed and dinosaur.x_axis > cactus.x_axis:
                    cactus.has_passed = True
                    obstacles += 1

            if len(dinosaurs) > 0:
                cactus.move()

            if cactus.x_axis + cactus.IMAGE.get_width() < 0:
                cacti_to_remove.append(cactus)

        if has_to_add_cactus:
            new_cactus_x_axis = SCREEN_WIDTH + SCREEN_WIDTH * random.choice(reference_values_to_set_new_x_axis)
            cacti.append(Cactus(new_cactus_x_axis, speed))

            if ai_playing:
                for genome in genomes_list:
                    genome.fitness += 5

        for cactus in cacti_to_remove:
            cacti.remove(cactus)

        render_screen(screen, ground, score, dinosaurs, cacti, speed, obstacles)


def run(config_path):
    ai_config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(ai_config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    if ai_playing:
        population.run(main, 50)
    else:
        main(None, None)


if __name__ == '__main__':
    run('config.txt')
