import requests
import time

metrics = {'cluster-filter':        ['gm-cluster',
                                     'onpremise(dongjak)',
                                     'mec(ilsan)',
                                     'innogrid-k8s-master',
                                     'innogrid-k8s-2',
                                     'innogrid-1',
                                     'ai-networklab(pangyo)',
                                     '5G-opentestlab(Daejeon)'],

           'kind': {'cluster':      ['cpu_util',
                                     'cpu_usage',
                                     'cpu_total',
                                     'memory_util',
                                     'memory_usage',
                                     'memory_total',
                                     'disk_util',
                                     'disk_usage',
                                     'disk_total',
                                     'pod_running',
                                     'pod_quota',
                                     'pod_util'],

                    'pod':          ['pod_cpu',
                                     'pod_memory',
                                     'pod_net_bytes_transmitted',
                                     'pod_net_bytes_received'],

                    'app':          ['pod_count',
                                     'sevice_count',  # Check spell sevice -> service?
                                     'deployment_count',
                                     'cronjob_count',
                                     'job_count',
                                     'pv_count',
                                     'pvc_count',
                                     'namespace_count'],

                    'gpu':          ['gpu_tempreture',
                                     'gpu_power',
                                     'gpu_power_limit',
                                     'gpu_memory_total',
                                     'gpu_memory_used',
                                     'gpu_memory_free',
                                     'gpu_ratio',
                                     'gpu_memory_ratio',
                                     'gpu_fan_speed'],

                    'namespace':    ['namespace_cpu',
                                     'namespace_memory',
                                     'namespace_pod_count']
                    }
          }


def concat_metrics(metric_index):
    ret = ""
    for i in metric_index:
        ret += cluster_metrics[i]
        ret += '|'
    
    ret = ret[:-1]
    print(ret)

def validate_requset(cluster, metric):
    if cluster not in clusters:
        return False
    
    if not set(metric.split('_')) <= set(cluster_metrics):
        return False
    
    return True

def requset_monitor(ip, port, kind, cluster, metric, auth):
    if validate_requset(cluster, metric):
        t = int(time.time())
        uri = 'http://{}:{}/kube/v1/monitoring/{}?start={}&end={}&step=1s&cluster_filter={}&metric_filter={}'.format(ip, port, kind, t, t, cluster, metric)
        print(uri)

        return requests.get(uri, auth=auth)
    else:
        return False

if __name__ == '__main__':
    res = requset_monitor('gm-cluster', 'cpu_util', auth)

    if res:
        res_json = res.json()
        print(res_json)
    
    else:
        pass

