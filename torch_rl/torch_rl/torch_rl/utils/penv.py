from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "initial_reset":
            for i in range(data):
                _ = env.reset()
            obs = env.reset()
            conn.send(obs)
        elif cmd == "get_env_seed":
            env_seed = env.get_env_seed()
            conn.send(env_seed)
        elif cmd == "add_dro_seed":
            env.add_dro_seed(data)
        elif cmd == "remove_dro_seed":
            env.remove_dro_seed(data)
        elif cmd == "get_dro_seeds":
            dro_seeds = env.get_dro_seeds()
            conn.send(dro_seeds)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
    
    def initial_reset(self):
        for i, local in enumerate(self.locals):
            local.send(("initial_reset", (i+1)*50))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results
    
    def get_env_seeds(self):
        for local in self.locals:
            local.send(("get_env_seed", None))
        results = [self.envs[0].get_env_seed()] + [local.recv() for local in self.locals]
        return results

    def add_dro_seed(self, seed):
        for local in self.locals:
            local.send(("add_dro_seed", seed))
        self.envs[0].add_dro_seed(seed)

    def remove_dro_seed(self, seed):
        for local in self.locals:
            local.send(("remove_dro_seed", seed))
        self.envs[0].remove_dro_seed(seed)

    def get_dro_seeds(self):
        for local in self.locals:
            local.send(("get_dro_seeds", None))
        results = [self.envs[0].get_dro_seeds()] + [local.recv() for local in self.locals]
        return results

    def render(self):
        raise NotImplementedError