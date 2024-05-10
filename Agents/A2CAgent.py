from Agents.AgentUtils import *
from Models.A2CModel import A2CModel
from Models.ModelUtils import loadModel, saveModel, copyModel

class A2CAgent:
    def __init__(self, agent_cfg: Dict[str, Any]):

        self.policy_model = A2CModel()
        if agent_cfg['load_model']:
            print(f"Loading model from {agent_cfg['load_model']}")
            loadModel(self.policy_model, agent_cfg['load_model'])
        self.policy_model.to(device)
        self.target_model = A2CModel()
        copyModel(self.policy_model, self.target_model)
        self.target_model.to(device)

        self.policy_model.train()
        self.target_model.eval()

        self.optimizer_value = torch.optim.Adam(self.policy_model.getValueParams(), lr=agent_cfg['value_lr'])
        self.optimizer_policy = torch.optim.Adam(self.policy_model.getPolicyParams(), lr=agent_cfg['policy_lr'])

        self.loss_func = nn.MSELoss()

        self.epsilon = agent_cfg['epsilon']
        self.gamma = agent_cfg['gamma']


    def update(self, batch_data: TensorTuple) -> Tuple[float, float, float]:
        """
        A2C算法的更新过程
        :param batch_data: [states, actions, rewards, next_states, terminate_flags]
        :return:
        """
        states, actions, rewards, next_states, terminate_flags = batch_data
        # states: (B, 6, 33, 33)
        # actions: (B,)
        # rewards: (B,)
        # next_states: (B, 6, 33, 33)
        # terminate_flags: (B,)

        # TD_target = r + gamma * V(s')
        with torch.no_grad():
            _, V_s_next = self.target_model(next_states)
            td_target = rewards + self.gamma * V_s_next * (1 - terminate_flags)

        # TD_error = TD_target - V(s)
        policy, V_s = self.policy_model(states)
        policy = torch.distributions.Categorical(policy)
        entropy = policy.entropy()

        advantage = (td_target - V_s).detach()
        policy_loss = (-policy.log_prob(actions) * advantage - 0.01 * entropy).mean()

        value_loss = self.loss_func(V_s, td_target) * 10

        total_loss = policy_loss + value_loss

        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        total_loss.backward()
        self.optimizer_policy.step()
        self.optimizer_value.step()

        return policy_loss.item(), value_loss.item(), total_loss.item()


    def updateTargetModel(self):
        copyModel(self.policy_model, self.target_model)

    @torch.no_grad()
    def getAction(self, state: State) -> Action:
        if random.random() < self.epsilon:
            return Action.randomAction()
        else:
            policy, _ = self.policy_model(state.tensor.unsqueeze(0))
            return Action(policy.argmax())


    def train(self):
        self.policy_model.train()

    def eval(self):
        self.policy_model.eval()

    def save(self, model_path: str):
        saveModel(self.policy_model, model_path)
