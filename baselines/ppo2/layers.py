import numpy as np
import tensorflow as tf

from baselines.a2c.utils import ortho_init, lstm, lnlstm
from baselines.common.models import register, nature_cnn


class RNN(object):
    def __init__(self, func, memory_size=None, agent_outputs=None):
        self._func = func
        self.memory_size = memory_size
        self.agent_outputs = agent_outputs

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


@register("ppo_lstm")
def ppo_lstm(num_units=128, layer_norm=False):
    def network_fn(input, mask, state):
        input = tf.layers.flatten(input)
        mask = tf.to_float(mask)

        if layer_norm:
            h, next_state = lnlstm([input], [mask[:, None]], state, scope='lnlstm', nh=num_units)
        else:
            h, next_state = lstm([input], [mask[:, None]], state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(network_fn, memory_size=num_units * 2)


@register("ppo_cnn_lstm")
def ppo_cnn_lstm(num_units=128, layer_norm=False, **conv_kwargs):
    def network_fn(input, mask, state):
        mask = tf.to_float(mask)
        initializer = ortho_init(np.sqrt(2))

        h = nature_cnn(input, **conv_kwargs)
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, units=512, activation=tf.nn.relu, kernel_initializer=initializer)

        if layer_norm:
            h, next_state = lnlstm([h], [mask[:, None]], state, scope='lnlstm', nh=num_units)
        else:
            h, next_state = lstm([h], [mask[:, None]], state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(network_fn, memory_size=num_units * 2)


@register("ppo_cnn_lnlstm")
def ppo_cnn_lnlstm(num_units=128, **conv_kwargs):
    return ppo_cnn_lstm(num_units, layer_norm=True, **conv_kwargs)


@register("qmix")
def qmix(num_units=128, num_agents=3, embed_dim=32, num_actions=10, num_output_dims=32):
    def network_fn(input, mask, state, is_training=False):
        obs = input['obs']
        mask = tf.to_float(mask)

        obs_embed_net = tf.make_template('obs_embed_network',
                                         _unit_embed_net,
                                         layer_norm=True,
                                         num_hidden=num_units)

        agent_net = tf.make_template('agent_network',
                                     _lstm_agent,
                                     obs_embed_net=obs_embed_net,
                                     num_hidden=num_units)

        hidden = []
        next_state = []

        for i in range(num_agents):
            h, next_state_i = agent_net(obs=obs[:, i],
                                        mask=mask,
                                        state=state[:, i * num_units * 2: (i + 1) * num_units * 2])
            hidden.append(h)
            next_state.append(next_state_i)

        hidden = tf.stack(hidden, axis=1)
        next_state = tf.concat(next_state, axis=1)

        return hidden, next_state

    return RNN(network_fn, memory_size=num_units * 2 * num_agents)

@register("qmix_value_network")
def qmix_value_network(num_units=128, num_agents=3, embed_dim=32, num_actions=10, num_output_dims=32):
    # def network_fn(input, mask, state, is_training=False):
    def network_fn(obs_states,
                   n_agents=3,
                   agent_dim=5,
                   n_enemy=4,
                   enemy_dim=4,
                   n_actions=10,
                   num_units=256):
        ally_states = obs_states[:, :n_agents * agent_dim]
        enemy_states = obs_states[:, n_agents * agent_dim: n_agents * agent_dim + n_enemy * enemy_dim]
        last_actions = obs_states[:, n_agents * agent_dim + n_enemy * enemy_dim:]

        ally_states = tf.reshape(ally_states, shape=(-1, n_agents, agent_dim))
        enemy_states =tf.reshape(enemy_states, shape=(-1, n_enemy, enemy_dim))
        last_actions = tf.reshape(last_actions, shape=(-1, n_agents, n_actions))
        ally_states = tf.concat([ally_states, last_actions], axis=2)

        ally_embed_net = tf.make_template('ally_embed_network',
                                          _unit_embed_net,
                                            num_hidden=num_units,
                                            layer_norm=True)

        enemy_embed_net = tf.make_template('enemy_embed_network',
                                           _unit_embed_net,
                                           num_hidden=num_units,
                                           layer_norm=True)

        ally_embeds = ally_embed_net(ally_states)
        enemy_embeds = enemy_embed_net(enemy_states)

        ally_embeds = max_pool(ally_embeds, ally_states)
        enemy_embeds = max_pool(enemy_embeds, enemy_states)

        states_embeds = tf.concat([ally_embeds, enemy_embeds], axis=1)
        embed = tf.layers.dense(states_embeds, num_units, tf.nn.relu, True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        embed = tf.layers.dense(embed, 1, None, True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        return tf.squeeze(embed, axis=1)
    return RNN(network_fn, memory_size=num_units * 2 * num_agents)



def _unit_embed_net(unit_embed,
                    activation_fn=tf.nn.relu,
                    num_hidden=64,
                    layer_norm=False,
                    ):
    embed = tf.layers.dense(unit_embed, num_hidden, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    embed = tf.layers.dense(embed, num_hidden, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    if layer_norm:
        embed = tf.contrib.layers.layer_norm(embed, center=True, scale=True)

    return embed


def _type_embed_net(unit_embed,
                    activation_fn=tf.nn.relu,
                    num_hidden=64,
                    num_output_dims=64,
                    layer_norm=False):
    embed = tf.layers.dense(unit_embed, num_hidden, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    if layer_norm:
        embed = tf.contrib.layers.layer_norm(embed, center=True, scale=True)
    return embed


def _lstm_agent(obs,
                mask,
                state,
                obs_embed_net,
                num_hidden=64,
                ):
    obs_embeds = obs_embed_net(obs)
    h, next_state = lstm([obs_embeds], [mask[:, None]], state, scope='lstm', nh=num_hidden)
    h = h[0]
    return h, next_state


@register("openai_five")
def openai_five(num_units=256, num_output_dims=64, layer_norm=False):
    def network_fn(input, mask, state, num_hidden=num_units, is_training=False):
        player_embed_net = tf.make_template('player_type_embed_network',
                                            _type_embed_net,
                                            num_hidden=num_units,
                                            num_output_dims=num_output_dims,
                                            layer_norm=True)
        neutral_embed_net = tf.make_template('neutral_type_embed_network',
                                             _type_embed_net,
                                             num_hidden=num_units,
                                             num_output_dims=num_output_dims,
                                             layer_norm=True)
        enemy_embed_net = tf.make_template('enemy_type_embed_network',
                                           _type_embed_net,
                                           num_hidden=num_units,
                                           num_output_dims=num_output_dims,
                                           layer_norm=True)

        unit_embed_net = tf.make_template('unit_embed_network', _unit_embed_net,
                                          num_hidden=num_units,
                                          num_output_dims=num_output_dims,
                                          layer_norm=True)

        player_units = input['feature_units_self']
        neutral_units = input['feature_units_neutral']
        enemy_units = input['feature_units_enemy']

        player_embeds = player_embed_net(unit_embed_net(player_units))
        neutral_embeds = neutral_embed_net(unit_embed_net(neutral_units))
        enemy_embeds = enemy_embed_net(unit_embed_net(enemy_units))

        player_embeds = max_pool(player_embeds, player_units)
        neutral_embeds = max_pool(neutral_embeds, neutral_units)
        enemy_embeds = max_pool(enemy_embeds, enemy_units)

        embeds = tf.concat([player_embeds, neutral_embeds, enemy_embeds], axis=1)
        embeds = tf.layers.dense(embeds, num_hidden, tf.nn.relu, True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        embeds = tf.contrib.layers.layer_norm(embeds, center=True, scale=True)

        h = embeds
        h, next_state = lstm([h], [tf.to_float(mask)[:, None]], state, scope='lstm', nh=num_units)
        h = h[0]
        return h, next_state

    return RNN(network_fn, memory_size=num_units * 2)


def max_pool(unit_embeds, units):
    unit_embeds = tf.expand_dims(unit_embeds, axis=1)
    unit_embeds = tf.nn.max_pool(unit_embeds, [1, 1, units.get_shape()[1], 1], [1, 1, 1, 1],
                                 padding='VALID')
    unit_embeds = tf.squeeze(unit_embeds, axis=[1, 2])
    unit_embeds = tf.layers.flatten(unit_embeds)
    return unit_embeds


def _lstm_embed_net(input, sequence_length, num_hidden=64, only_last_output=True):
    output, state = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden),
        input,
        dtype=tf.float32,
        sequence_length=sequence_length,
    )

    if only_last_output:
        output = output[:, -1]
        return output
    else:
        return output, state


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    return length


def used(sequence):
    return tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
