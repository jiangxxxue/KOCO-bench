## Ascend社区评论命令一览

Ascend社区中所有项目均由Bot维护，这意味着开发人员可以在每个Pull Request或者Issue下面通过评论触发Bot命令。

<table class="command">
    <thead>
        <tr>
            <th>命令</th>
            <th width="25%">示例</th>
            <th>描述</th>
            <th>面向对象</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                /check-cla
            </td>
            <td style="white-space:nowrap;">
                /check-cla
            </td>
            <td>
                强制重新检查Pull Request的CLA状态。
                如果Pull Request的提交者已经签署了CLA协议，则<strong>ascend-cla/yes</strong>标签将会被添加到Pull Request中；如果没有，则标签<strong>ascend-cla/no</strong>将被添加到Pull Request中。
            </td>
            <td>
                所有开发者
            </td>
        </tr>
        <tr>
            <td>
                /cla cancel
            </td>
            <td style="white-space:nowrap;">
                /cla cancel
            </td>
            <td>
                强制删除<strong>ascend-cla/yes</strong>标签。
            </td>
            <td>
               Repo_Admins，仓库维护者，Pull Request提交者
            </td>
        </tr>
        <tr>
           <td>
                /compile
           </td>
           <td style="white-space:nowrap;">
                /compile
           </td>
           <td>
                触发编译流水线。
                编译通过后，该Pull Request会被打上<strong>ci-pipeline-passed</strong>的标签。若编译失败，该Pull Request会被打上<strong>ci-pipeline-failed</strong>的标签。
           </td>
           <td>
              所有开发者
           </td>
        </tr>
        <tr>
            <td>
                /lgtm [cancel]
            </td>
            <td style="white-space:nowrap;">
                /lgtm
                <br/>
                /lgtm cancel
            </td>
            <td>
                添加或移除用于代表代码已经评审过的标签 <strong>lgtm</strong>。      
            </td>
            <td>
              Repo_Admins，仓库维护者，或者仓库Committers
            </td>
        </tr>
        <tr>
            <td>
                /approve [cancel]
            </td>
            <td style="white-space:nowrap;">
                /approve
                <br/>
                /approve cancel
            </td>
            <td>
                添加或移除用于代表同意合并的标签<strong>approved</strong>。
            </td>
            <td>
              Repo_Admins，仓库维护者，或者Committers
            </td>
        </tr>
        <tr>
            <td>
                /close
            </td>
            <td style="white-space:nowrap;">
                /close
            </td>
            <td>
               关闭Pull Request或者Issue。
            </td>
            <td>
              Repo_Admins，仓库维护者，或者Committers
            </td>
        </tr>
        <tr>
            <td>
                /reopen
            </td>
            <td style="white-space:nowrap;">
                /reopen
            </td>
            <td>
                重新打开Pull Request或者Issue。
            </td>
            <td>
              Repo_Admins，仓库维护者，或者Committers
            </td>
        </tr>
        <tr>
            <td>
                /retest
            </td>
            <td style="white-space:nowrap;">
                /retest
            </td>
            <td>
                重新运行已失败的测试作业。
            </td>
            <td>
                任何人都可以在Pull Request上触发此命令。
            </td>
        </tr>
        <tr>
            <td>
                /check-pr
            </td>
            <td style="white-space:nowrap;">
                /check-pr
            </td>
            <td>
                检查Pull Request中的标签是否满足条件，如果满足条件，则合并Pull Request。
            </td>
            <td>
                任何人都可以在Pull Request上触发此命令。
            </td>
        </tr>
        <tr>
            <td>
                /squash
            </td>
            <td style="white-space:nowrap;">
                /squash
            </td>
            <td>
                更改Pull Request中的merge方法为squash merge。
                squash merge：该分支中的多个提交被打包成一个提交到目标分支。
            </td>
            <td>
              Repo_Admins，仓库维护者，或者Committers
            </td>
        </tr>
        <tr>
            <td>
                /squash cancel
            </td>
            <td style="white-space:nowrap;">
                /squash cancel
            </td>
            <td>
                删除 <strong>merge/squash</strong>标签。
            </td>
            <td>
              Repo_Admins，仓库维护者，或者Committers
            </td>
        </tr>
    </tbody>
</table>