#include "config.h"
#include "option_list.h"

list *make_data_config(int num_config, char *configs[][2]) {
    list *options = make_list();
    int i;
    for (i = 0; i < num_config; i++) {
        option_insert(options, configs[i][0], configs[i][1]);
    }
    return options;
}

/**
 * ネットワーク構成の設定情報を作成する
 *
 * 設定例: darknet のネットワーク設定ファイルの内容を以下のような構造で表現する
 * int num_types = 5;  // typesの要素数
 * char *types[] = {
 *       "[net]",
 *       "[connected]",
 *       "[dropout]",
 *       "[connected]",
 *       "[softmax]"
 * };
 * int num_configs[5] = {16, 18, 19, 21, 22};  // n番目の設定はconfigs[num_congis[n]] まで
 * char *configs[][2] = {
 *       // [net]
 *       {"batch", "1"},
 *       {"subdivisions", "1"},
 *       {"height", "4"},
 *       {"width", "4"},
 *       {"channels", "3"},
 *       {"max_crop", "4"},
 *       {"min_crop", "4"},
 *       {"hue", ".1"},
 *       {"saturation", ".75"},
 *       {"exposure", ".75"},
 *       {"learning_rate", "0.1"},
 *       {"policy", "poly"},
 *       {"power", "4"},
 *       {"max_batches", "100"},
 *       {"momentum", "0.9"},
 *       {"decay", "0.0005"},
 *       // [connected]
 *       {"output", "10"},
 *       {"activation", "relu"},
 *       // [dropout]
 *       {"probability", ".5"},
 *       // [connected]
 *       {"output", "3"},
 *       {"activation", "linear"},
 *       // [softmax]
 *        {"groups", "1"}
 * };
 *
 * @param num_types
 * @param types
 * @param num_configs
 * @param configs
 * @return ネットワーク構成の設定
 * @memo 返り値の解放は呼び出しで行う必要がある
 */
list *make_network_config(int num_types, char **types, int *num_configs, char *configs[][2]) {
    list *options = make_list();
    section *current = NULL;
    int i = 0;
    int j = 0;
    for (i = 0; i < num_types; i++) {
        current = malloc(sizeof(section));
        current->type = types[i];  // TODO copy する
        current->options = make_list();
        for (; j < num_configs[i]; j++) {
            option_insert(current->options, configs[j][0], configs[j][1]);
        }
        list_insert(options, current);
    }

    return options;
}